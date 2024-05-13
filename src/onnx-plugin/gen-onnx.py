# 生成插件onnx
# // @dong 整理归档 2024.5

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import torch.autograd
import os
from torch.onnx import symbolic_helper

class MYSELUImpl(torch.autograd.Function):

    # reference: https://pytorch.org/docs/1.10/onnx.html#torch-autograd-functions
    @staticmethod
    def symbolic(g, x, p):
        print("==================================call symbolic")
        return g.op("MYSELU", x, p, 
            g.op("Constant", value_t=torch.tensor([3, 2, 1], dtype=torch.float32)),
            attr1_s="这是字符串属性", 
            attr2_i=[1, 2, 3], 
            attr3_f=222
        )

    @staticmethod
    def forward(ctx, x, p):
        return x / (1 + torch.exp(-x))   #可以导出
        #return x * 1 / (1 + torch.exp(-x))  #不可以
        #return x * 2/ (1 + torch.exp(-x))   #可以



class MYSELU(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.param = nn.parameter.Parameter(torch.arange(n).float())

    def forward(self, x):
        return MYSELUImpl.apply(x, self.param)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(1, 1, 3, padding=1)

        self.conv.weight.data.fill_(1)
        self.conv.bias.data.fill_(0)
        self.myselu = MYSELU(3)

    def forward(self, x):
        x = self.conv(x)
        #x = self.myselu(x)
        #x = self.selu_torch(x)
        x = self.myselu(x)
        return x

# import torch.onnx.symbolic_opset11
print("对应opset文件夹代码在这里：", os.path.dirname(torch.onnx.__file__))

model = Model().eval()
input = torch.tensor([
    # batch 0
    [
        [1,   1,   1],
        [1,   1,   1],
        [1,   1,   1],
    ],
        # batch 1
    [
        [-1,   1,   1],
        [1,   0,   1],
        [1,   1,   -1]
    ]
], dtype=torch.float32).view(2, 1, 3, 3)



output = model(input)
print(f"inference output = \n{output}")

dummy = torch.zeros(1, 1, 3, 3)

torch.onnx.export(
    model, 

    # 这里的args，是指输入给model的参数，需要传递tuple，因此用括号
    (dummy,), 

    # 储存的文件路径
    "workspace/demo.onnx", 

    # 打印详细信息
    verbose=True, 

    # 为输入和输出节点指定名称，方便后面查看或者操作
    input_names=["image"], 
    output_names=["output"], 

    # 这里的opset，指，各类算子以何种方式导出，对应于symbolic_opset11
    opset_version=14, 

    operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
    # 表示他有batch、height、width3个维度是动态的，在onnx中给其赋值为-1
    # 通常，我们只设置batch为动态，其他的避免动态
    dynamic_axes={
        "image": {0: "batch", 2: "height", 3: "width"},
        "output": {0: "batch", 2: "height", 3: "width"},
    },

    # 对于插件，需要禁用onnx检查
    #enable_onnx_checker=False

)

print("Done.!")


