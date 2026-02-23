import torch
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz

def implement_integrated(model: torch.nn.Module,
                         valid_loader: torch.utils.data.DataLoader,
                         device: torch.device):

    model.eval()
    model.to(device)

    images, labels = next(iter(valid_loader))

    input_tensor = images[0].unsqueeze(0).to(device)
    baseline = torch.zeros_like(input_tensor).to(device)

    ig = IntegratedGradients(model)

    with torch.no_grad():
        output = model(input_tensor)
        target_class = torch.argmax(output).item()

    attributions, delta = ig.attribute(
        input_tensor,
        baseline,
        target=target_class,
        return_convergence_delta=True
    )

    attr = attributions.squeeze().cpu().detach().numpy()
    attr = np.transpose(attr, (1, 2, 0))
    original_img = input_tensor.squeeze().cpu().detach().numpy()
    original_img = np.transpose(original_img, (1, 2, 0))
    viz.visualize_image_attr(
        attr,
        original_img,
        method="blended_heat_map",
        sign="positive",
        show_colorbar=True
    )

    plt.show()

    print("Target class:", target_class)
    print("Convergence delta:", delta.item())