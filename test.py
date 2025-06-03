import torch
from model_s3d import S3D

# Load both
ft1 = torch.load("checkpoints/finetuning/s3d_partial_ft_epoch01.pt")["model_state_dict"]
ft2 = torch.load("checkpoints/finetuning/epoch07.pt")["model_state_dict"]

# Compare weights
# print(torch.equal(ft1["s3d.base.14.branch1.0.weight"], ft2["s3d.base.14.branch1.0.weight"]))

# print(list(ft1.keys())[:50])  # Or adjust the number
# print(list(ft2.keys())[:50])

# s3d = S3D(400)
#
# for name, _ in s3d.named_parameters():
#     print(name)


print(torch.equal(ft1["s3d.base.9.branch1.1.conv_t.weight"], ft2["s3d.base.9.branch1.1.conv_t.weight"]))
