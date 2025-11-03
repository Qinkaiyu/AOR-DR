# Imports
import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import models_vit
import torch.nn.functional as F
import argparse
import sys
import random
import os.path as osp
from torchmetrics.functional import pairwise_manhattan_distance
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import label_binarize
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import multiprocessing
from diffloss import DiffLoss
from timm.models.layers import trunc_normal_
from pos_embed import interpolate_pos_embed
import math
from einops import rearrange, repeat
import torchvision.models as models
import timm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib import cm

class SimpleFusion(nn.Module):
    def __init__(self, feature_dim=768, context_dim=1, emb_dim=768):
        super(SimpleFusion, self).__init__()
        self.position_embedding = nn.Embedding(4, emb_dim)

        #affine fusion
        self.proj_feature = nn.Linear(feature_dim, emb_dim)
        self.proj_context = nn.Embedding(5, emb_dim)
        # cross-attn
        # self.proj_context = nn.Linear(context_dim, emb_dim)
        # self.proj_context = nn.Embedding(5, emb_dim)
    def forward(self, features, context,t):
        '''
        :param features: [batch_size, 1024]
        :param context: [batch_size, 5]
        :return: [batch_size, 1024]
        '''
        # 投影到嵌入维度
        # affine fusion
        # features = self.proj_feature(features)  # [batch_size, emb_dim]\
        # context = self.proj_context(context.long())  # [batch_size, emb_dim]
        # context = context.squeeze(1)
        # position = torch.tensor(t).repeat(context.shape[0],1)
        # position = position.squeeze(1).to(device='cuda')
        # position_embedding = self.position_embedding(position)
        # context = context + position_embedding
        # out = context * features + features  # [batch_size, emb_dim]
        # cross-attn 
        features = self.proj_feature(features)  # [batch_size, emb_dim]
        context = self.proj_context(context.long())  # [batch_size, emb_dim]
        position = torch.tensor(t).repeat(context.shape[0],1).to(device='cuda')
        position_embedding = self.position_embedding(position)
        position_embedding = position_embedding.squeeze(1)
        context = context.squeeze(1)
        context = context + position_embedding
        attention_score = torch.matmul(features,context.transpose(0,1))
        attention_weight = F.softmax(attention_score,dim=1)
        attention_context = torch.matmul(attention_weight,context)
        out = attention_context + features  
        return out

def task_importance_weights(label_array):
    uniq = torch.unique(label_array)
    num_examples = label_array.size(0)
    m = torch.zeros(uniq.shape[0])

    for i, t in enumerate(torch.arange(torch.min(uniq), torch.max(uniq))):
        m_k = torch.max(torch.tensor([label_array[label_array > t].size(0),
                                      num_examples - label_array[label_array > t].size(0)]))
        m[i] = torch.sqrt(m_k.float())

    imp = m / torch.max(m)
    return imp
class RegressionDataset(Dataset):
    def __init__(self, images_root, data_file, transforms):
        self.images_root = images_root
        self.labels = []
        self.images_file = []
        self.transforms = transforms
        with open(data_file) as fin:
            for line in fin:
                # image_file, image_label = line.split()
                splits = line.split()
                image_file = splits[0]
                labels = splits[1:]
                self.labels.append([int(label) for label in labels])
                self.images_file.append(image_file)
        self.name = osp.splitext(osp.basename(data_file))[0].lower()
        if "val" in self.name or "test" in self.name:
            print(f"Dataset prepare: val/test data_file: {data_file}")
        elif "train" in self.name:
            print(f"Dataset prepare: train data_file: {data_file}")
        else:
            raise ValueError(f"Invalid data_file: {data_file}")
        print(f"Dataset prepare: len of labels: {len(self.labels[0])}")
        print(f"Dataset prepare: len of dataset: {len(self.labels)}")
    def __getitem__(self, index):
        img_file, target_list = self.images_file[index], self.labels[index]
        if "val" in self.name or "test" in self.name:
            target = target_list[len(target_list) // 2]
        else:
            target = random.choice(target_list)

        full_file = os.path.join(self.images_root, img_file)
        img = Image.open(full_file)
        if img.mode == "L":
            img = img.convert("RGB")
        if self.transforms:
            img = self.transforms(img)
        return img, target
    def __len__(self):
        return len(self.labels)
def prepare_model(chkpt_dir, arch='vit_large_patch16'):
    # build model
    model = models_vit.__dict__[arch](
        img_size=224,
        num_classes=5,
        drop_path_rate=0,
        global_pool=True,
    )
    # load model
    print('-------------------------------vit-------------------------',model)
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    checkpoint_model = checkpoint['model']
    interpolate_pos_embed(model, checkpoint_model)

    msg = model.load_state_dict(checkpoint_model, strict=False)
    assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
    trunc_normal_(model.head.weight, std=2e-5)

    print('-------------------------------model-------------------------',msg)
    return model
def prepare_model_vit(arch='vit_base_patch16_224', pretrained=True):
    # 使用 timm 加载预训练的 ViT-B 模型
    model = timm.create_model(arch, pretrained=pretrained)
    print('model',model)
    # 打印模型信息
    print('-------------------------------vit-------------------------', model)

    return model
class ViTFeatureExtractor(nn.Module):
    def __init__(self, vit_model,
                 decoder_embed_dim=768,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 diffloss_d=3,
                 diffloss_w=1024,
                 num_sampling_steps='100',
                 diffusion_batch_mul=4,
                 grad_checkpointing=False,
                 ):
        super(ViTFeatureExtractor, self).__init__()
        self.vit_model = vit_model
        self.head = self.vit_model.head
        self.token_embed_dim = 1
        self.diffloss = DiffLoss(
            target_channels=self.token_embed_dim,
            z_channels=decoder_embed_dim,
            width=diffloss_w,
            depth=diffloss_d,
            num_sampling_steps=num_sampling_steps,
            grad_checkpointing=grad_checkpointing
        )
        self.diffusion_batch_mul = diffusion_batch_mul
        self.SimpleFusion = SimpleFusion()
    def forward(self, image,p,target,t):
        target = target.unsqueeze(1)
        target = target.repeat(self.diffusion_batch_mul, 1)
        if t == 0:
            x = self.vit_model.forward_features(image.float())
            # for vit_base_patch16_224
            x = x[:,0,:]
            z = x.repeat(self.diffusion_batch_mul, 1)
            loss, probas = self.diffloss(z=z, target=target)
            fusion_feature = z
        if t !=0:
            p = torch.round(p)
            p = torch.clamp(p, 0, 1)  # 限制在 0 和 1 之间
            fusion_feature  = self.SimpleFusion(image,p,t)
            loss, probas = self.diffloss(z=fusion_feature, target=target)
        return loss, probas,fusion_feature
    def sample(self,image,t,temperature=1.0,cfg=1.0):
        if t == 0:
            x = self.vit_model.forward_features(image.float())
            # for vit_base_patch16_224
            x = x[:,0,:]
            # x.requires_grad = True
        else:
            x = image
        sampled_token_latent = self.diffloss.sample(x,temperature=1.0,cfg=1.0)
        return sampled_token_latent,x
class ResnetFeatureExtractor(nn.Module):
    def __init__(self, model,
                 decoder_embed_dim=2048,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 diffloss_d=3,
                 diffloss_w=1024,
                 num_sampling_steps='100',
                 diffusion_batch_mul=4,
                 grad_checkpointing=False,
                 ):
        super(ResnetFeatureExtractor, self).__init__()
        self.model = model
        self.token_embed_dim = 1
        self.diffloss = DiffLoss(
            target_channels=self.token_embed_dim,
            z_channels=decoder_embed_dim,
            width=diffloss_w,
            depth=diffloss_d,
            num_sampling_steps=num_sampling_steps,
            grad_checkpointing=grad_checkpointing
        )
        self.diffusion_batch_mul = diffusion_batch_mul
        self.SimpleFusion = SimpleFusion()
    def forward(self, image,p,target,t):
        target = target.unsqueeze(1)
        target = target.repeat(self.diffusion_batch_mul, 1)
        if t == 0:
            x = self.model(image.float())
            x = x.squeeze()  # 去掉所有大小为 1 的维度
            print('x',x.shape)
            
            z = x.repeat(self.diffusion_batch_mul, 1)
            loss, probas = self.diffloss(z=z, target=target)
            fusion_feature = z
        if t !=0:
            fusion_feature  = self.SimpleFusion(image,p)
            loss, probas = self.diffloss(z=fusion_feature, target=target)
        return loss, probas,fusion_feature
    def sample(self,image,t,temperature=1.0,cfg=1.0):
        if t == 0:
            x = self.model(image.float())
            x = x.squeeze()  # 去掉所有大小为 1 的维度
        else:
            x = image
        return self.diffloss.sample(x,temperature=1.0,cfg=1.0),x
def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def label_to_vector(labels, max_length=4):
    '''
    :param labels: [batch_size]，包含标签的张量
    :param max_length: 向量的最大长度
    :return: [batch_size, max_length]，转换后的向量
    '''
    batch_size = labels.size(0)
    vectors = torch.zeros(batch_size, max_length, dtype=torch.int)

    for i in range(batch_size):
        label = labels[i].item()
        vectors[i, :label] = 1

    return vectors
def cohen_quadratic_kappa(preds, gt, num_classes):
    """
    计算模型预测和 GT 的 Cohen's Quadratic Kappa
    :param preds: 模型预测结果 (1D list or array)
    :param gt: 真实标签 (1D list or array)
    :param num_classes: 分类总数
    :return: Cohen's Quadratic Kappa 值
    """
    gt = np.array(gt).astype(int)  # 确保 gt 是整数数组
    preds = np.array(preds).astype(int)  # 确保 pred 是整数数组
    # 构造混淆矩阵
    confusion_matrix = np.zeros((num_classes, num_classes))
    for p, g in zip(preds, gt):
        confusion_matrix[g][p] += 1

    # 创建二次加权矩阵
    weights = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            weights[i][j] = (i - j) ** 2 / (num_classes - 1) ** 2

    # 计算观察值 (Observed Agreement)
    observed_agreement = confusion_matrix / np.sum(confusion_matrix)

    # 计算期望值 (Expected Agreement)
    hist_gt = np.sum(confusion_matrix, axis=1)  # GT 的分布
    hist_preds = np.sum(confusion_matrix, axis=0)  # 模型预测的分布
    expected_agreement = np.outer(hist_gt, hist_preds) / np.sum(confusion_matrix)

    # 加权求和
    observed_score = np.sum(weights * observed_agreement)
    expected_score = np.sum(weights * (expected_agreement / np.sum(confusion_matrix)))

    # 计算 Kappa 值
    kappa = 1 - (observed_score / expected_score)
    return kappa

def main():
    vit_base_patch16_224 = prepare_model_vit()
    feature_extractor = ViTFeatureExtractor(vit_base_patch16_224)
    # vit_large_patch16 = prepare_model(chkpt_dir=r'./RETFound_cfp_weights.pth')
    # feature_extractor = ViTFeatureExtractor(vit_large_patch16)
    # feature_extractor.load_state_dict(torch.load('./vit-b_messidor_position.pth'))
    feature_extractor.to(device = 'cuda')
    for param in feature_extractor.parameters():
        param.requires_grad = True
    # for param in feature_extractor.head.parameters():
    #     param.requires_grad = True
    
    num_trainable_params = count_trainable_parameters(feature_extractor)
    print(f"可训练参数的数量: {num_trainable_params}")
    custom_transform = transforms.Compose([transforms.Resize((224, 224)),
                                           transforms.RandomHorizontalFlip(),  # 随机水平翻转
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_dataset = RegressionDataset(images_root=r'C:\Users\yuqinkai\Downloads\GDRBench_AllowedShare\FundusDG_mini\images\messidor',
                                      data_file=r'C:\Users\yuqinkai\Downloads\GDRBench_AllowedShare\FundusDG_mini\splits\MESSIDOR_train.txt',
                                      transforms=custom_transform)

    custom_transform2 = transforms.Compose([transforms.Resize((224, 224)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])])

    # test_dataset = RegressionDataset(images_root=r'C:\Users\yuqinkai\Downloads\GDRBench_AllowedShare\FundusDG_mini\images',
    #                                  data_file=r'C:\Users\yuqinkai\Downloads\GDRBench_AllowedShare\FundusDG_mini\splits\EYEPACS_train.txt',
    #                                  transforms=custom_transform2)

    valid_dataset1 = RegressionDataset(images_root=r'C:\Users\yuqinkai\Downloads\GDRBench_AllowedShare\FundusDG_mini\images\messidor',
                                      data_file=r'C:\Users\yuqinkai\Downloads\GDRBench_AllowedShare\FundusDG_mini\splits\MESSIDOR_crossval.txt',
                                      transforms=custom_transform2)
    # valid_dataset2 = RegressionDataset(images_root=r'C:\Users\yuqinkai\Downloads\GDRBench_AllowedShare\FundusDG_mini\images',
    #                                   data_file=r'C:\Users\yuqinkai\Downloads\GDRBench_AllowedShare\FundusDG_mini\splits\APTOS_crossval.txt',
    #                                   transforms=custom_transform2)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=64,
                              shuffle=True,
                              num_workers=8)
    valid_loader1 = DataLoader(dataset=valid_dataset1,
                              batch_size=64,
                              shuffle=True,
                              num_workers=8)
    best_accuracy1 = 0
    best_accuracy2 = 0
    eff_batch_size = 64
    lr = 1e-4
    optimizer = torch.optim.AdamW(feature_extractor.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=True)
    print('-------------------------------train-------------------------')
    for epoch in range(300):
        feature_extractor.train()
        total_cost = 0
        for batch_idx, (image, targets) in enumerate(train_loader):
            image = image.to(device = 'cuda')
            targets = targets.to(device = 'cuda')
            total_loss = 0
            vectors_targets = label_to_vector(targets)
            vectors_targets = vectors_targets.T.float().to(device = 'cuda')
            for t in range(4):
                if t == 0:
                    initial_p = torch.zeros_like(vectors_targets[t]).unsqueeze(1)
                    loss,p,fusion_feature = feature_extractor(image, initial_p,vectors_targets[t],t)
                else:
                    loss,p,fusion_feature = feature_extractor(fusion_feature, p,vectors_targets[t],t)
                    weight= 1
                    loss = loss * weight
                total_loss += loss
            print('total_loss',total_loss)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            total_cost += total_loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_cost / len(train_loader)}")
        with open('vit-b_aptos.txt', 'a') as f:
            f.write(f'Epoch {epoch}, Average Loss: {total_cost / len(train_loader):.4f}\n')
        print('-------------------------------valid-------------------------')
        # 测试
        feature_extractor.eval()
        total_cost = 0
        correct = 0
        total = 0
        total_cost = 0
        correct = 0
        total = 0
        all_pred = []
        all_targets = []
        for image, targets in valid_loader1:
            image = image.to(device = 'cuda')
            targets = targets.to(device = 'cuda')
            with torch.no_grad():
                all_probas = []
                for t in range(4):
                    if t == 0:
                        probas,features = feature_extractor.sample(image,t,temperature=1.0,cfg=1.0,)
                        fusion_feature  = feature_extractor.SimpleFusion(features,probas,t)
                        all_probas.append(probas)
                    else:
                        probas,fusion_feature = feature_extractor.sample(fusion_feature,t,temperature=1.0,cfg=1.0)
                        fusion_feature  = feature_extractor.SimpleFusion(fusion_feature,probas,t)
                        all_probas.append(probas)
                all_probas = [torch.round(p) for p in all_probas]
                all_probas_tensor = torch.stack(all_probas).squeeze(-1)  # 变成 [64, 5]
                all_probas_tensor = torch.stack(all_probas).transpose(0, 1)  # 变成 [64, 5]
                pred = torch.sum(all_probas_tensor, dim=1)
                pred = pred.T
                print('pred',pred)
                print('targets',targets)
                all_pred.append(pred.squeeze().cpu().numpy())
                all_targets.append(targets.squeeze().cpu().numpy())
                total += targets.size(0)
                correct += pred.eq(targets).sum().item()
            accuracy2 = correct / total
        all_pred = np.concatenate(all_pred, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        cm = confusion_matrix(all_targets, all_pred)
        print('cm',cm)

        sensityvity = []
        specificity = []
        for i in range(5):
            tn = cm.sum()-(cm[i,:].sum()+cm[:,i].sum()+cm[i,i])
            fp = cm[i,:].sum()-cm[i,i]
            fn = cm[:,i].sum()-cm[i,i]
            tp = cm[i,i]
            sensitivity_i = tp/(tp+fn) if (tp+fn) > 0 else 0
            specificity_i = tn/(tn+fp) if (tn+fp) > 0 else 0
            sensityvity.append(sensitivity_i)
            specificity.append(specificity_i)
        for i in range(5):
            print('sensitivity',sensityvity[i])
            print('specificity',specificity[i])
        print('average sensitivity',np.mean(sensityvity))
        print('average specificity',np.mean(specificity))
        f1 = f1_score(all_targets, all_pred, average='macro')

        print('-------------------------------accuracy-------------------------')
        print('accuracy',accuracy2)
        with open('vit-b_messidor.txt', 'a') as f:
            f.write(f'Epoch {epoch}, accuracy: {accuracy2:.4f},f1: {f1:.4f},sensitivity: {np.mean(sensityvity):.4f},specificity: {np.mean(specificity):.4f}\n')
        if accuracy2 > best_accuracy2:
            best_accuracy2 = accuracy2
            torch.save(feature_extractor.state_dict(), './vit-b_messidor_position.pth')
            print(f"Saved new best model with accuracy: {best_accuracy2:.4f}")
        # scheduler.step(accuracy)
    print('best_accuracy2',best_accuracy2)
    # print('best_accuracy1',best_accuracy1)
def test():
    random.seed(42)
    vit_base_patch16_224 = prepare_model_vit()
    feature_extractor = ViTFeatureExtractor(vit_base_patch16_224)
    feature_extractor.load_state_dict(torch.load('./vit-b_aptos_position_cross.pth',map_location='cuda:0'))
    # vit_large_patch16 = prepare_model(chkpt_dir=r'./RETFound_cfp_weights.pth')
    # feature_extractor = ViTFeatureExtractor(vit_large_patch16)
    feature_extractor.to(device = 'cuda')
    for param in feature_extractor.parameters():
        param.requires_grad = True
    # for param in feature_extractor.head.parameters():
    #     param.requires_grad = True
    
# 在模型初始化后调用
    num_trainable_params = count_trainable_parameters(feature_extractor)
    print(f"可训练参数的数量: {num_trainable_params}")

    custom_transform2 = transforms.Compose([transforms.Resize((224, 224)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])])

    # test_dataset = RegressionDataset(images_root=r'C:\Users\yuqinkai\Downloads\GDRBench_AllowedShare\FundusDG_mini\images',
    #                                  data_file=r'C:\Users\yuqinkai\Downloads\GDRBench_AllowedShare\FundusDG_mini\splits\EYEPACS_train.txt',
    #                                  transforms=custom_transform2)


    valid_dataset2 = RegressionDataset(images_root=r'C:\Users\yuqinkai\Downloads\GDRBench_AllowedShare\FundusDG_mini\images',
                                      data_file=r'C:\Users\yuqinkai\Downloads\GDRBench_AllowedShare\FundusDG_mini\splits\APTOS_crossval.txt',
                                      transforms=custom_transform2)


    valid_loader2 = DataLoader(dataset=valid_dataset2,
                              batch_size=64,
                              shuffle=True,
                              num_workers=8)
    # test_loader = DataLoader(dataset=test_dataset,
    #                          batch_size=64,
    #                          shuffle=False,
    #                          num_workers=8)
    best_accuracy1 = 0
    best_accuracy2 = 0
    eff_batch_size = 64
    # lr = 5e-4 * eff_batch_size / 256
    lr = 1e-4
    optimizer = torch.optim.AdamW(feature_extractor.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=True)
    print('-------------------------------valid-------------------------')
        # 测试
    feature_extractor.train()
    total_cost = 0
    correct = 0
    total = 0
    f1 = 0
    k = 0
    fusion_feature_list1 = []
    fusion_feature_list2 = []
    fusion_feature_list3 = []
    fusion_feature_list4 = []
    fusion_feature0 = []
    targets2_list = []
    label1 = []
    label2 = []
    label3 = []
    label4 = []
    all_pred = []
    all_targets = []
    for image2,targets2 in valid_loader2:
        image2 = image2.to(device = 'cuda')
        targets2 = targets2.to(device = 'cuda')
        # targets2_list = targets2.squeeze().cpu().numpy()
        targets2_list.append(targets2)
        with torch.no_grad():
            all_probas = []
            for t in range(4):
                if t == 0:
                    probas,features = feature_extractor.sample(image2,t,temperature=1.0,cfg=1.0)
                    fusion_feature  = feature_extractor.SimpleFusion(features,probas,t)
                    all_probas.append(probas)
                    
                else:
                    probas,fusion_feature = feature_extractor.sample(fusion_feature,t,temperature=1.0,cfg=1.0)
                    fusion_feature  = feature_extractor.SimpleFusion(fusion_feature,probas,t)  
                    all_probas.append(probas)
            all_probas = [torch.round(p) for p in all_probas]
            all_probas_tensor = torch.stack(all_probas).squeeze(-1)  # 变成 [64, 5]
            all_probas_tensor = torch.stack(all_probas).transpose(0, 1)  # 变成 [64, 5]
            pred = torch.sum(all_probas_tensor, dim=1)
            pred = pred.T
            print('pred',pred)
            print('targets',targets2)
            print('pred',pred.shape)
            print('targets',targets2.shape)
            total += targets2.size(0)
            correct += pred.eq(targets2).sum().item()
            pred = pred.squeeze().cpu().numpy()
            targets2 = targets2.squeeze().cpu().numpy()
            f1 += f1_score(targets2, pred, average='macro')
            print('f1',f1)
            k += cohen_quadratic_kappa(pred, targets2, 5)
            print('k',k)
            all_pred.append(pred)
            all_targets.append(targets2)
    all_pred = np.concatenate(all_pred, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    cm = confusion_matrix(all_targets, all_pred)
    print('cm',cm)

    sensityvity = []
    specificity = []
    for i in range(5):
        tn = cm.sum()-(cm[i,:].sum()+cm[:,i].sum()+cm[i,i])
        fp = cm[i,:].sum()-cm[i,i]
        fn = cm[:,i].sum()-cm[i,i]
        tp = cm[i,i]
        sensitivity_i = tp/(tp+fn) if (tp+fn) > 0 else 0
        specificity_i = tn/(tn+fp) if (tn+fp) > 0 else 0
        sensityvity.append(sensitivity_i)
        specificity.append(specificity_i)
    for i in range(5):
        print('sensitivity',sensityvity[i])
        print('specificity',specificity[i])
    print('average sensitivity',np.mean(sensityvity))
    print('average specificity',np.mean(specificity))
    accuracy1 = correct / total
    f1 = f1 / len(valid_loader2)
    k = k / len(valid_loader2)
    print('-------------------------------accuracy-------------------------')
    print('accuracy',accuracy1)
    print('f1',f1)
    print('k',k)
if __name__ == '__main__':
    main()
    #test()


