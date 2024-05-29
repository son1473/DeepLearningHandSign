import torchvision.models as models

# 모델 불러오기
mobilenet_v2 = models.mobilenet_v2(pretrained=True)
mobilenet_v3_small = models.mobilenet_v3_small(pretrained=True)
efficientnet_b0 = models.efficientnet_b0(pretrained=True)
efficientnet_b1 = models.efficientnet_b1(pretrained=True)
squeezenet1_0 = models.squeezenet1_0(pretrained=True)
squeezenet1_1 = models.squeezenet1_1(pretrained=True)
shufflenet_v2_x0_5 = models.shufflenet_v2_x0_5(pretrained=True)
mnasnet0_5 = models.mnasnet0_5(pretrained=True)
mnasnet1_0 = models.mnasnet1_0(pretrained=True)
resnet18 = models.resnet18(pretrained=True)

# -----------------------------------------------------

import torch
import torchvision.models as models
import torch.optim as optim

# 사전 학습된 가중치를 사용하는 ResNet18 모델
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 학습률 설정
learning_rate = 0.001

# Adam Optimizer
adam_optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# AdamW Optimizer
adamw_optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# RMSProp Optimizer
rmsprop_optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

# Nesterov Accelerated Gradient (NAG) Optimizer
nag_optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)

# SGD Optimizer
sgd_optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Optimizers 리스트로 관리
optimizers = {
    "adam": adam_optimizer,
    "adamw": adamw_optimizer,
    "rmsprop": rmsprop_optimizer,
    "nag": nag_optimizer,
    "sgd": sgd_optimizer
}

# Optimizer 사용 예시
for name, optimizer in optimizers.items():
    print(f"Using optimizer: {name}")
    # 여기에 학습 루프를 넣어 각 optimizer로 모델을 학습시킵니다.
    # 예시로 1번의 forward-backward 과정을 보여줍니다.
    model.train()
    optimizer.zero_grad()
    # 가상의 데이터와 타겟 (실제 학습 데이터와 타겟으로 대체해야 합니다)
    inputs = torch.randn(64, 3, 64, 64)  # 배치 크기 64, 채널 3, 이미지 크기 64x64
    targets = torch.randint(0, 10, (64,))  # 10개의 클래스로 가정
    outputs = model(inputs)
    loss = torch.nn.functional.cross_entropy(outputs, targets)
    loss.backward()
    optimizer.step()
