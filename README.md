# 🧠 Memory Lane - Helping Alzheimer’s Patients Remember

## 🎥 Demo Videos

Watch these videos to see Memory Lane in action:

- [Memory Lane Feature Walkthrough](https://youtu.be/cyOLuWBITWQ)
- [Memory Lane Edge Cases](https://youtu.be/MoXNAE0Pd4c)

## ⚙️ Installation

1. **Clone the repository** to your local machine:

   ```bash
   git clone https://github.com/Spiral-Memory/MemoryLane.git
   ```

2. **Navigate to the project directory**:

   ```bash
   cd MemoryLane
   ```

3. **Ensure Python 3.10.0** is installed to avoid dependency errors.

4. **Create a virtual environment**:

   ```bash
   python3 -m venv .venv
   ```

5. **Activate the virtual environment**:

   - On Windows:

     ```bash
     .venv\Scripts\activate
     ```

6. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## 📥 Installing Models

1. **Download `intent_cf_model`**, extract it, and place it in the root folder:
   [Download intent_cf_model](https://drive.google.com/drive/folders/1w6HQQCWSCbliR0Y_rR7_B3CdueFDFzlg?usp=sharing)

2. **Download `vosk-model-small-en-in-0.4`** for speech-to-text and place it in the root folder:
   [Download vosk-model-small-en-in-0.4](https://alphacephei.com/vosk/models/vosk-model-small-en-in-0.4.zip)

## 🧪 Creating Models for Testing

To generate the `intent_cf_model` manually:

```bash
cd dev_test/intent_train
python train.py
```

## 🗄️ Setting Up the Database

1. Create a `.env` file and add your MongoDB connection string:

   ```
   DB_URL=mongodb+srv://username:password@cluster_url/?retryWrites=true&w=majority&appName=appName
   ```

## 🚀 Usage

1. Head to `main.py`
2. Run the main script:

   ```bash
   python main.py
   ```

The script launches an **Assistance Bot** for user interaction.

### Options Menu

3. Choose an action by entering the corresponding number:

   - **1**: Add a new relative
   - **2**: Start recognition
   - **3**: Exit the program

### Adding a New Relative

- Enter name, address, relationship, and gender
- Upload relative’s images
- The system generates facial embeddings and updates the detector

### Starting Recognition

- Choose between:

  - **Voice Mode** (Option 1)
  - **Text Mode** (Option 2)

The bot will assist accordingly.

### Exiting the Program

- Select option **3** to exit gracefully

### Invalid Input Handling

- On invalid input, you’ll see: **"Invalid Choice"**
- The bot will prompt you again

_Just follow the terminal prompts, and the bot will walk you through everything smoothly._
