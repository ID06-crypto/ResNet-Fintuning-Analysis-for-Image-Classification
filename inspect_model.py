from transformers import ResNetForImageClassification

model_name = "microsoft/resnet-18"
model = ResNetForImageClassification.from_pretrained(model_name)

print("Model Modules:")
for name, module in model.named_modules():
    if "conv" in name:
        print(name)
