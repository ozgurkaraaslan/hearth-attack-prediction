import tkinter as tk
from tkinter import messagebox
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler


class LogRegModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(14, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        logits = self.relu(x)
        return logits


model = LogRegModel()
model.load_state_dict(torch.load("model.pth"))
model.eval()


class HeartDiseasePredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Heart Disease Predictor")

        self.entries = {}
        self.fields = ["Age", "Cigs Per Day", "Tot Chol", "Sys BP", "Dia BP", "BMI", "Heart Rate", "Glucose"]
        self.data_test1 = {
            "Sex": "Male",
            "Age": 46,
            "Current Smoker": "No",
            "Cigs Per Day": 0,
            "BP Meds": "No",
            "Prevalent Stroke": "No",
            "Prevalent Hyp": "No",
            "Diabetes": "No",
            "Tot Chol": 250,
            "Sys BP": 121,
            "Dia BP": 81,
            "BMI": 28.73,
            "Heart Rate": 95,
            "Glucose": 76,
        }
        self.data_test2 = {
            "Sex": "Male",
            "Age": 40,
            "Current Smoker": "Yes",
            "Cigs Per Day": 20,
            "BP Meds": "No",
            "Prevalent Stroke": "No",
            "Prevalent Hyp": "No",
            "Diabetes": "No",
            "Tot Chol": 205,
            "Sys BP": 158,
            "Dia BP": 102,
            "BMI": 25.45,
            "Heart Rate": 75,
            "Glucose": 87,
        }

        self.create_radio_button("Sex", ["Male", "Female"], 0)
        self.create_entry_field("Age", 1)
        self.create_radio_button("Current Smoker", ["Yes", "No"], 2)
        self.create_entry_field("Cigs Per Day", 3)
        self.create_radio_button("BP Meds", ["Yes", "No"], 4)
        self.create_radio_button("Prevalent Stroke", ["Yes", "No"], 5)
        self.create_radio_button("Prevalent Hyp", ["Yes", "No"], 6)
        self.create_radio_button("Diabetes", ["Yes", "No"], 7)
        self.create_entry_field("Tot Chol", 8)
        self.create_entry_field("Sys BP", 9)
        self.create_entry_field("Dia BP", 10)
        self.create_entry_field("BMI", 11)
        self.create_entry_field("Heart Rate", 12)
        self.create_entry_field("Glucose", 13)

        self.test_button1 = tk.Button(root, text="Test-1", command=lambda: self.fill_with_data(self.data_test1))
        self.test_button1.grid(row=14, column=0, pady=10)

        self.test_button2 = tk.Button(root, text="Test-2", command=lambda: self.fill_with_data(self.data_test2))
        self.test_button2.grid(row=14, column=1, pady=10)

        self.predict_button = tk.Button(root, text="Predict", command=self.predict)
        self.predict_button.grid(row=14, column=2, pady=10)

    def create_radio_button(self, field, options, row):
        label = tk.Label(self.root, text=field)
        label.grid(row=row, column=0, padx=10, pady=5)
        var = tk.StringVar(value="")
        for idx, option in enumerate(options):
            rb = tk.Radiobutton(self.root, text=option, variable=var, value=option)
            rb.grid(row=row, column=idx + 1, padx=10, pady=5)
        self.entries[field] = var

    def create_entry_field(self, field, row):
        label = tk.Label(self.root, text=field)
        label.grid(row=row, column=0, padx=10, pady=5)
        entry = tk.Entry(self.root)
        entry.grid(row=row, column=1, padx=10, pady=5)
        self.entries[field] = entry

    def fill_with_data(self, data):
        for field, value in data.items():
            if field in ["Sex", "Current Smoker", "BP Meds", "Prevalent Stroke", "Prevalent Hyp", "Diabetes"]:
                self.entries[field].set(value)
            else:
                self.entries[field].delete(0, tk.END)
                self.entries[field].insert(0, value)

    def predict(self):
        try:
            data = []
            data.append(1 if self.entries["Sex"].get() == "Male" else 0)
            data.append(float(self.entries["Age"].get()))
            data.append(1 if self.entries["Current Smoker"].get() == "Yes" else 0)
            data.append(float(self.entries["Cigs Per Day"].get()))
            data.append(1 if self.entries["BP Meds"].get() == "Yes" else 0)
            data.append(1 if self.entries["Prevalent Stroke"].get() == "Yes" else 0)
            data.append(1 if self.entries["Prevalent Hyp"].get() == "Yes" else 0)
            data.append(1 if self.entries["Diabetes"].get() == "Yes" else 0)
            data.append(float(self.entries["Tot Chol"].get()))
            data.append(float(self.entries["Sys BP"].get()))
            data.append(float(self.entries["Dia BP"].get()))
            data.append(float(self.entries["BMI"].get()))
            data.append(float(self.entries["Heart Rate"].get()))
            data.append(float(self.entries["Glucose"].get()))

            input_data = np.array([data])
            sc = StandardScaler()
            input_data = sc.fit_transform(input_data)
            input_data_tensor = torch.tensor(input_data, dtype=torch.float32)

            with torch.no_grad():
                prediction = model(input_data_tensor)
                predicted_class = torch.argmax(prediction, dim=1)

                if predicted_class.item() == 0:
                    result = f"Result: Positive"
                else:
                    result = f"Result: Negative"
            messagebox.showinfo("Prediction Result", result)
        except Exception as e:
            messagebox.showerror("Input Error", str(e))


root = tk.Tk()
app = HeartDiseasePredictorApp(root)
root.mainloop()
