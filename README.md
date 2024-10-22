# Health Status Prediction Project

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
This project is a **Health Status Prediction System** that predicts the health condition of individuals based on input symptoms using machine learning models like **Decision Tree** and **Random Forest**. The frontend is developed using **Streamlit**, making the system interactive and easy to use.

## Features
- Prediction of health status based on symptoms.
- Machine Learning models: **Decision Tree** and **Random Forest**.
- Web-based interface built using **Streamlit**.
- Easy-to-use interface for health predictions.

## Technologies Used
- **Python**: Programming language used for building the machine learning models.
- **Scikit-learn**: Machine learning library used for training and testing the models.
- **Pandas**: Used for data manipulation and analysis.
- **Streamlit**: Framework used for building the web application frontend.
- **Joblib**: Used for saving and loading trained models.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/sahebjeet0313/health-status-prediction.git
    cd health-status-prediction
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

## Usage

1. After running the Streamlit app, open your browser and navigate to:
    ```
    http://localhost:8501
    ```

2. Enter the required symptoms in the form provided.

3. Click on the "Predict" button to get the predicted health status based on the symptoms provided.

## Model Performance
The system uses two models for prediction:
- **Decision Tree**: A tree-based algorithm for classification.
- **Random Forest**: An ensemble learning method that constructs multiple decision trees to improve accuracy.

Model evaluation results:
- **Accuracy (Decision Tree)**: 85%
- **Accuracy (Random Forest)**: 90%

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
