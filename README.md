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

3. Install the required dependencies using pip:

    ```bash
    pip install -r requirements.txt
    ```

## Generating the Intent Classifier Model

1. Locate the `train.py` file in the project directory.
2. Run the file to generate the `intent_cf_model` model, which the project will use for intent classification:

    ```bash
    python train.py
    ```

## Setting Up the Project

1. Navigate to the main script file, `main.py`.
2. Run the script to start the project:

    ```bash
    python main.py
    ```

## Usage

The main script, `main.py`, launches an Assistance Bot that helps users interact with the system. Follow these steps:

1. Run the script:

    ```bash
    python main.py
    ```

2. Upon starting, you’ll see a welcome message and a menu of options.

3. Select an option by typing the corresponding number and pressing Enter:
    - **1**: Add a new relative.
    - **2**: Start recognition.
    - **3**: Exit the program.

4. **Adding a New Relative**:
    - If you choose option `1`, you’ll be prompted to:
        - Enter the name, address, relationship, and gender of the relative.
        - Upload images of the relative as instructed.
    - The script will then generate embeddings and update the face detector.

5. **Starting Recognition**:
    - If you choose option `2`, select a mode:
        - **1**: Voice mode for voice recognition.
        - **2**: Text mode for text recognition.
    - The bot will operate in the selected mode to assist with recognition tasks.

6. **Exiting the Program**:
    - Choosing option `3` will safely exit the application.

7. **Invalid Input**:
    - If an invalid option is entered, the bot will display "Invalid Choice" and prompt you to select again.

Follow the terminal prompts for smooth interaction with the Assistance Bot.
