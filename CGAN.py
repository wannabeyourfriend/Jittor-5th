import jittor as jt
from jittor import init
import argparse
import os
import numpy as np
import math
from jittor import nn

if jt.has_cuda:
    jt.flags.use_cuda = 1

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=128, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0003, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--latent_dim', type=int, default=150, help='dimensionality of the latent space')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--n_classes', type=int, default=10, help='number of classes for dataset')
parser.add_argument('--img_size', type=int, default=64, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=1000, help='interval between image sampling')
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # 增加embedding维度以提供更丰富的类别信息
        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes * 2)
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                # 调整BatchNorm的momentum参数
                layers.append(nn.BatchNorm1d(out_feat, 0.9))
            # 使用更大的LeakyReLU斜率
            layers.append(nn.LeakyReLU(0.3))
            return layers
            
        # 增加网络深度和宽度
        self.model = nn.Sequential(
            *block((opt.latent_dim + opt.n_classes * 2), 256, normalize=False),
            *block(256, 512),
            *block(512, 1024),
            *block(1024, 2048),
            nn.Linear(2048, int(np.prod(img_shape))),
            nn.Tanh()
        )
        
    def execute(self, noise, labels):
        gen_input = jt.contrib.concat((self.label_emb(labels), noise), dim=1)
        img = self.model(gen_input)
        # 将img从1024维向量变为32*32矩阵
        img = img.view((img.shape[0], *img_shape))
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # 增加embedding维度
        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes * 2)
        
        self.model = nn.Sequential(
            nn.Linear(opt.n_classes * 2 + int(np.prod(img_shape)), 1024),
            nn.LeakyReLU(0.3),
            nn.Linear(1024, 512),
            nn.Dropout(0.5),  # 增加dropout比例
            nn.LeakyReLU(0.3),
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 添加Sigmoid激活函数
        )

    def execute(self, img, labels):
        d_in = jt.contrib.concat((img.view((img.shape[0], (- 1))), self.label_embedding(labels)), dim=1)
        output = self.model(d_in)
        return output
        # TODO: 将d_in输入到模型中并返回计算结果

# 损失函数：平方误差
# 调用方法：adversarial_loss(网络输出A, 分类标签B)
# 计算结果：(A-B)^2
adversarial_loss = nn.MSELoss()

generator = Generator()
discriminator = Discriminator()

# 导入MNIST数据集
from jittor.dataset.mnist import MNIST
import jittor.transform as transform
transform = transform.Compose([
    transform.Resize(opt.img_size),
    transform.Gray(),
    # 添加数据增强
    transform.RandomRotation(10),  # 随机旋转
    transform.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
    transform.ImageNormalize(mean=[0.5], std=[0.5]),
])
dataloader = MNIST(train=True, transform=transform).set_attrs(batch_size=opt.batch_size, shuffle=True)

optimizer_G = nn.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = nn.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

from PIL import Image
def save_image(img, path, nrow=10, padding=5):
    N,C,W,H = img.shape
    if (N%nrow!=0):
        print("N%nrow!=0")
        return
    ncol=int(N/nrow)
    img_all = []
    for i in range(ncol):
        img_ = []
        for j in range(nrow):
            img_.append(img[i*nrow+j])
            img_.append(np.zeros((C,W,padding)))
        img_all.append(np.concatenate(img_, 2))
        img_all.append(np.zeros((C,padding,img_all[0].shape[2])))
    img = np.concatenate(img_all, 1)
    img = np.concatenate([np.zeros((C,padding,img.shape[2])), img], 1)
    img = np.concatenate([np.zeros((C,img.shape[1],padding)), img], 2)
    min_=img.min()
    max_=img.max()
    img=(img-min_)/(max_-min_)*255
    img=img.transpose((1,2,0))
    if C==3:
        img = img[:,:,::-1]
    elif C==1:
        img = img[:,:,0]
    Image.fromarray(np.uint8(img)).save(path)

def sample_image(n_row, batches_done):
    # 随机采样输入并保存生成的图片
    z = jt.array(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))).float32().stop_grad()
    labels = jt.array(np.array([num for _ in range(n_row) for num in range(n_row)])).float32().stop_grad()
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.numpy(), "%d.png" % batches_done, nrow=n_row)

# ----------
#  模型训练
# ----------

# 添加早停机制
best_fid = float('inf')  # 使用FID分数作为评价指标
patience = 10  # 容忍轮数
patience_counter = 0

for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.shape[0]

        # 数据标签，valid=1表示真实的图片，fake=0表示生成的图片
        valid = jt.ones([batch_size, 1]).float32().stop_grad()
        fake = jt.zeros([batch_size, 1]).float32().stop_grad()

        # 真实图片及其类别
        real_imgs = jt.array(imgs)
        labels = jt.array(labels)

        # -----------------
        #  训练生成器
        # -----------------

        # 采样随机噪声和数字类别作为生成器输入
        z = jt.array(np.random.normal(0, 1, (batch_size, opt.latent_dim))).float32()
        gen_labels = jt.array(np.random.randint(0, opt.n_classes, batch_size)).float32()

        # 生成一组图片
        gen_imgs = generator(z, gen_labels)
        # 损失函数衡量生成器欺骗判别器的能力，即希望判别器将生成图片分类为valid
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)
        g_loss.sync()
        optimizer_G.step(g_loss)

        # ---------------------
        #  训练判别器
        # ---------------------

        validity_real = discriminator(real_imgs, labels)
        d_real_loss = adversarial_loss(validity_real,valid)#"""TODO: 计算真实类别的损失函数"""

        validity_fake = discriminator(gen_imgs.stop_grad(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake,fake)#"""TODO: 计算虚假类别的损失函数""")

        # 总的判别器损失
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.sync()
        optimizer_D.step(d_loss)
        if i  % 50 == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.data, g_loss.data)
            )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=10, batches_done=batches_done)

    # 每个epoch结束后评估模型
    if epoch % 5 == 0:  # 每5个epoch评估一次
        current_fid = evaluate_fid(generator)  # 需要实现FID评估函数
        
        if current_fid < best_fid:
            best_fid = current_fid
            # 保存最佳模型
            generator.save("generator_best.pkl")
            discriminator.save("discriminator_best.pkl")
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
            
    generator.save("generator_last.pkl")
    discriminator.save("discriminator_last.pkl")

generator.eval()
discriminator.eval()
generator.load('generator_last.pkl')
discriminator.load('discriminator_last.pkl')

number = '20249782020202'
#写入要生成的字符串
n_row = len(number)
z = jt.array(np.random.normal(0, 1, (n_row, opt.latent_dim))).float32().stop_grad()
labels = jt.array(np.array([int(number[num]) for num in range(n_row)])).float32().stop_grad()
gen_imgs = generator(z,labels)

img_array = gen_imgs.data.transpose((1,2,0,3))[0].reshape((gen_imgs.shape[2], -1))
min_=img_array.min()
max_=img_array.max()
img_array=(img_array-min_)/(max_-min_)*255
Image.fromarray(np.uint8(img_array)).save("result2.png")