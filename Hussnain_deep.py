import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),                  
    transforms.ToTensor(),                        
    transforms.Normalize((0.5,), (0.5,))          
])

class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.conv1 = nn.Conv2d(1,32,kernel_size = 3,padding = 1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32,64,kernel_size = 3,padding = 1)

        self.fc1 = nn.Linear(64*7*7,128)
        self.fc2 = nn.Linear(128,10)

    def forward(self,x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.pool(out)
        
        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool(out)
        
        out = torch.flatten(out,1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        
        return out


def main():
    st.title('Digit Classifier')
    st.write('Upload any handwritten digit and see the output')

    file = st.file_uploader('Please upload an image', type=['png', 'jpg'])
    if file:
        img = Image.open(file).convert('L')
        st.image(img, use_container_width=True)

        img_tensor = transform(img).unsqueeze(0)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = DigitCNN().to(device)
        model.load_state_dict(torch.load(
            r"C:\Users\Husmiya\Streamlit\Digit_Classifier.pth",
            map_location=device
        ))
        model.eval()
        
        with torch.no_grad():
            output = model(img_tensor.to(device))
            predictions = torch.softmax(output, dim=1).cpu().numpy()[0]

        Digit_classes = [str(i) for i in range(10)]
        fig, ax = plt.subplots()
        y_pos = np.arange(len(Digit_classes))
        ax.barh(y_pos, predictions, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(Digit_classes)
        ax.invert_yaxis()
        ax.set_xlabel('Probability')
        ax.set_title('Digit Prediction')

        st.pyplot(fig)
    else:
        st.text('You have not uploaded the file yet')


if __name__ == '__main__':
    main()
