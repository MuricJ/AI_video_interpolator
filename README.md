# **AI Video Interpolator**

This repository contains an implementation of an easy-to-train video interpolator that is based on the FILM model and is made for the AI class project in the University of Coimbra

---

## **Getting Started**

Follow these instructions to get a copy of the project running on your local machine.

### **Prerequisites**
You have to acquire the following:
- Python 3.12
- Python libraries from requirements.txt
- vimeo90k dataset

### **Installation**

1. **Install dependencies:**
   - For Python:
     ```
     pip install -r requirements.txt
     ```
2. **Download the dataset:**
    By default, the vimeo90k dataset should be placed in a folder named "vimeo90k" within the src directory.

2. **Run the application:**
     ```bash
     python main.py
     ```
    By default, the program will start training. Currently here is not CLI interface, behaviour can be altered by calling the appropraite functoin at the bottom of the main.py file. It contains examples for inference and generating videos.

4. **Default configuration:**
     ```
        EPOCHS = 8
        BATCH_SIZE = 8
        LR = 8e-5
        CRITERION = losses.L1_only
        CHECKPOINT = None
     ```
---

## 📂 **Project Structure**
```
root/
├── src/
│   ├── vimeo90k
│   ├── checkpoints
├── requirements.txt
└── README.md
```