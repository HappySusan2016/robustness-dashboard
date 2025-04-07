# If running outside of a Streamlit environment, replace Streamlit UI with print statements or skip execution
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ModuleNotFoundError:
    print("Warning: Streamlit is not available. Running in non-Streamlit mode.")
    STREAMLIT_AVAILABLE = False

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import io

# Skip torch imports if not available
try:
    import torch
    import torchvision.transforms as transforms
    import torchvision.models as models
    import requests
    from torchvision.utils import save_image
    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    print("Warning: PyTorch and torchvision are not available. Adversarial examples will be skipped.")
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    # Function to generate adversarial example using FGSM
    def fgsm_attack(image, epsilon, data_grad):
        sign_data_grad = data_grad.sign()
        perturbed_image = image + epsilon * sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return perturbed_image

    # Load a pretrained model and set to eval mode
    model = models.resnet50(pretrained=True)
    model.eval()

    # Download and preprocess a sample chest X-ray image
    image_url = "https://upload.wikimedia.org/wikipedia/commons/8/88/Pediatric_chest_PA_2.jpg"
    response = requests.get(image_url)
    original_image = Image.open(io.BytesIO(response.content)).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    input_image = transform(original_image).unsqueeze(0)
    input_image.requires_grad = True

    # Forward pass
    output = model(input_image)
    label = output.max(1, keepdim=True)[1]

    # Calculate loss and backpropagate
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(output, label.squeeze())
    model.zero_grad()
    loss.backward()
    data_grad = input_image.grad.data

    # Generate adversarial image
    epsilon_adv = 0.03
    adv_image = fgsm_attack(input_image, epsilon_adv, data_grad)
else:
    input_image = None
    adv_image = None

# Sample accuracy data (fake example)
attack_types = ['FGSM', 'PGD', 'CW', 'AutoAttack']
epsilon_values = [0.01, 0.02, 0.03, 0.05, 0.1]
robustness_data = {
    attack: [np.clip(1.0 - e * np.random.uniform(8, 12), 0.2, 0.9) for e in epsilon_values]
    for attack in attack_types
}

# Sample confidence histogram (clean vs adversarial)
clean_conf = np.random.beta(8, 2, 1000)
adversarial_conf = np.random.beta(4, 6, 1000)

if STREAMLIT_AVAILABLE:
    st.title("AI Model Robustness Dashboard for Medical Imaging")

    # Sidebar inputs
    model_name = st.sidebar.selectbox("Select Model", ["ResNet-50", "UNet"])
    dataset = st.sidebar.selectbox("Select Dataset", ["Chest X-ray", "Lung CT", "Skin Lesions"])
    attack_selected = st.sidebar.selectbox("Select Attack", attack_types)
    epsilon_selected = st.sidebar.slider("Select Epsilon (Perturbation Strength)", 0.01, 0.1, 0.03, step=0.01)

    # Section 1: Accuracy plot
    st.subheader("Clean vs. Adversarial Accuracy")
    bar_data = pd.DataFrame({
        'Accuracy': [0.88, robustness_data[attack_selected][epsilon_values.index(epsilon_selected)]],
        'Type': ['Clean', 'Adversarial']
    })
    st.bar_chart(bar_data.set_index('Type'))

    # Section 2: Robustness curve
    st.subheader("Robustness Curve: Accuracy vs. Epsilon")
    fig, ax = plt.subplots()
    for attack in attack_types:
        ax.plot(epsilon_values, robustness_data[attack], label=attack)
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Accuracy")
    ax.set_title("Robustness Curve")
    ax.legend()
    st.pyplot(fig)

    # Helper to convert matplotlib fig to BytesIO PNG
    def fig_to_bytes(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        return buf

    st.download_button("Download Robustness Plot", data=fig_to_bytes(fig), file_name="robustness_curve.png", mime="image/png")

    # Section 3: Confidence histogram
    st.subheader("Confidence Score Distribution")
    fig2, ax2 = plt.subplots()
    ax2.hist(clean_conf, bins=20, alpha=0.6, label='Clean')
    ax2.hist(adversarial_conf, bins=20, alpha=0.6, label='Adversarial')
    ax2.set_xlabel("Confidence Score")
    ax2.set_ylabel("Frequency")
    ax2.legend()
    st.pyplot(fig2)
    st.download_button("Download Confidence Plot", data=fig_to_bytes(fig2), file_name="confidence_plot.png", mime="image/png")

    # Section 4: Example image comparison (real adversarial example)
    st.subheader("Adversarial Sample Viewer")
    if TORCH_AVAILABLE:
        st.text("Original vs. Adversarial Image")
        orig_np = input_image.squeeze().permute(1, 2, 0).detach().numpy()
        adv_np = adv_image.squeeze().permute(1, 2, 0).detach().numpy()
        st.image([orig_np, adv_np], caption=['Original', 'Adversarial'], width=256)
    else:
        st.warning("Torch not available. Adversarial sample viewer is disabled.")

    # Section 5: Runtime statistics
    st.subheader("Performance Metrics")
    st.write(pd.DataFrame({
        'Metric': ['Attack Runtime (s)', 'Memory Usage (MB)', 'Throughput (imgs/sec)'],
        'Value': [round(np.random.uniform(0.5, 1.5), 2), round(np.random.uniform(200, 400), 2), round(np.random.uniform(5, 10), 2)]
    }))
else:
    print("Dashboard cannot be rendered because Streamlit is not installed.")
    print("Example Data Summary:")
    print("\nAccuracy Data:")
    for atk in attack_types:
        print(f"{atk}: {robustness_data[atk]}")
    print("\nClean vs Adversarial Confidence Scores:")
    print(f"Clean mean: {np.mean(clean_conf):.2f}, Adversarial mean: {np.mean(adversarial_conf):.2f}")
