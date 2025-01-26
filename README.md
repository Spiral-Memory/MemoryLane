# Memory Lane - Helping Alzheimer’s Patients Remember

## Installation

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/Spiral-Memory/MemoryLane.git
    ```

2. Navigate to the project directory:

    ```bash
    cd MemoryLane
    ```
3. Ensure that Python 3.10.0 is installed on your system. This version is required to avoid any potential dependency resolution errors.

4. Create a virtual environment:

    ```bash
    python3 -m venv .venv
    ```

5. Activate the virtual environment:
    - On Windows:
        ```bash
        .venv\Scripts\activate
        ```

6. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Installing Models

1. Download the `intent_cf_model` from the provided link, extract it, and place it in the root folder.  
   [Download intent_cf_model](https://drive.google.com/drive/folders/1w6HQQCWSCbliR0Y_rR7_B3CdueFDFzlg?usp=sharing) 

2. Download the `vosk-model-small-en-in-0.4` (used for speech-to-text) from the provided link, and place it in the root folder.  
   [Download vosk-model-small-en-in-0.4](https://alphacephei.com/vosk/models/vosk-model-small-en-in-0.4.zip)

## Creating Models for Testing

The `intent_cf_model` can also be generated manually. To do so, navigate to the `dev_test` folder, then go to the `intent_train` folder, and run the `train.py` file:

```bash
cd dev_test/intent_train
python train.py
```

## Setting Up the Database

1. Create a `.env` file and include the following line with your MongoDB connection string:

    ```
    DB_URL=mongodb+srv://username:password@cluster_url/?retryWrites=true&w=majority&appName=appName
    ```

## Setting Up the Project

1. Navigate to the main script file, `main.py`.
2. Run the script to start the project:

    ```bash
    python main.py
    ```

## Usage

The main script, `main.py`, launches an Assistance Bot to help users interact with the system. Follow these steps:

1. Run the script:

    ```bash
    python main.py
    ```

2. Upon starting, you'll see a welcome message and a menu of options.

3. Select an option by typing the corresponding number and pressing Enter:
    - **1**: Add a new relative.
    - **2**: Start recognition.
    - **3**: Exit the program.

4. **Adding a New Relative**:
    - If you choose option `1`, you will be prompted to:
        - Enter the name, address, relationship, and gender of the relative.
        - Upload images of the relative as instructed.
    - The script will generate embeddings and update the face detector.

5. **Starting Recognition**:
    - If you choose option `2`, you will select a mode:
        - **1**: Voice mode for voice recognition.
        - **2**: Text mode for text recognition.
    - The bot will assist with recognition tasks based on the selected mode.

6. **Exiting the Program**:
    - Choose option `3` to safely exit the application.

7. **Invalid Input**:
    - If an invalid option is entered, the bot will display "Invalid Choice" and prompt you to select again.

Follow the terminal prompts for seamless interaction with the Assistance Bot.
