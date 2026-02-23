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
import tqdm
def implement_integrated(model: torch.nn.Module,
                         valid_loader: torch.utils.data.DataLoader,
                         device: torch.device,
                         class_names: list[str],
                         num_images: int = 5):

    model.eval()
    model.to(device)

    ig = IntegratedGradients(model)

    shown = 0

    for batch_images, batch_labels in tqdm.tqdm(valid_loader):

        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)

        for i in range(batch_images.size(0)):

            input_tensor = batch_images[i].unsqueeze(0)
            baseline = torch.zeros_like(input_tensor)

            output = model(input_tensor)
            pred_class = torch.argmax(output, dim=1).item()
            true_class = batch_labels[i].item()

            attributions, delta = ig.attribute(
                input_tensor,
                baseline,
                target=pred_class,
                return_convergence_delta=True
            )

            # Convert attribution → numpy (HWC)
            attr = attributions.squeeze().cpu().detach().numpy()
            attr = np.transpose(attr, (1, 2, 0))

            # Convert image → numpy (HWC)
            original_img = input_tensor.squeeze().cpu().detach().numpy()
            original_img = np.transpose(original_img, (1, 2, 0))

            viz.visualize_image_attr(
                attr,
                original_img,
                method="blended_heat_map",
                sign="positive",
                show_colorbar=True
            )

            plt.title(
                f"Predicted: {class_names[pred_class]} | "
                f"True: {class_names[true_class]}"
            )

            plt.show()

            shown += 1
            if shown >= num_images:
                return