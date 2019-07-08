import torch
import torch.nn as nn

class UNet(nn.Module):

    def __init__(self, n_ch,patch_height,patch_width):
        super().__init__()
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.conv_enter = nn.Conv2d(1,32,(3,3), padding=1)
        self.conv1 = nn.Conv2d(32, 32, (3,3), padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.conv2_1 = nn.Conv2d(32, 64, (3,3), padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, (3,3), padding=1)
        self.conv2_3 = nn.Conv2d(64, 32, (3,3), padding=1)
        self.conv3_1 = nn.Conv2d(64, 128, (3,3), padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, (3,3), padding=1)
        self.conv3_3 = nn.Conv2d(128, 256, (3,3), padding=1)
        self.conv4 = nn.Conv2d(256, 128, (3,3), padding=1)
        self.conv4_1 = nn.Conv2d(128, 64, (3,3), padding=1)
        self.conv_last = nn.Conv2d(32, 1, (1,1), padding=0)

        self.maxpool = nn.MaxPool2d(kernel_size =2)
        self.upsample1 = nn.ConvTranspose2d(256,128,kernel_size=2,stride=2)
        self.upsample2 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.upsample3 = nn.ConvTranspose2d(64,32,kernel_size=2,stride=2)


        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        #Down_layer 1
        x = self.conv_enter(x)  # ( 1 >> 32)
        x = self.relu(x)
        x = self.dropout(x)
        conv1 = self.conv1(x)  #(32>>32)
        x = self.relu(x)
        x = self.maxpool(conv1)

        #Down_layer2  ()
        x = self.conv2_1(x)   #(32 >> 64)
        x = self.relu(x)
        x = self.dropout(x)
        conv2 = self.conv2_2(x)  #(64 >> 64)
        x = self.relu(x)
        x = self.maxpool(conv2)

        #Down_layer3  ()
        x = self.conv3_1(x)  #(64 >> 128)
        x = self.relu(x)
        x = self.dropout(x)
        conv3 = self.conv3_2(x)  #(128 >> 128)
        x = self.relu(x)
        x = self.maxpool(conv3)
        x = self.conv3_3(x) #(128 >> 256)

        #UP_layer1
        x = self.upsample1(x) #(256 >128)
        x = torch.cat([x, conv3], dim=1)  # 128>256

        x = self.conv4(x)  #(256>>128)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv3_2(x)  #(128 >> 128)
        x = self.relu(x)
        #UP_layer2
        x = self.upsample2(x) #(128 > 64)
        x = torch.cat([x, conv2], dim=1) #64>128

        x = self.conv4_1(x)  #(128 >> 64)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2_2(x)  #(64 >> 64)
        x = self.relu(x)

        #UP_layer3
        x = self.upsample3(x) #(64 > 32)
        x = torch.cat([x, conv1], dim=1) #32>64

        x = self.conv2_3(x)  #(64 >> 32)
        x = self.relu(x)
        x = self.dropout(x)

        #out_layer
        out = self.conv_last(x)  #(32 >> 2)
        out = self.sigmoid(out)
        

        return out
