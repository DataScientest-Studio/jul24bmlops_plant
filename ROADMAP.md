# Summary
The purpose of this message is to outline the various expectations relating to the project and its deadlines. How does a project work as part of an MLops Engineer training course with DataScientest?
Just follow these few steps.

## Step 0
[X] Framing (Our first meeting): 01.08.2024
- Meeting and presentation of the team
- Presentation of the project schedule and expectations
- Planning meetings and future actions

## Step 1
[ ] Building the specifications and the API: 09.08.2024
The most important phase of any software engineering project (including AI-based software) is to understand the business problem and create requirements. These requirements are translated into model objectives and model outputs. Possible errors and minimum launch success must be specified. To frame the requirements, you will need to draw up a specification following these guidelines.
Before diving in headlong on your own, don't hesitate to get together to discuss the following points:
- Define the context for your project
- Identify the specific process that could be fed by AI/ML
- Break this process down into a succession of tasks
- Identify the tasks where human actions can be automated
- You can use this to help you: Machine Learning Canvas template (v1.1)
Rendering(s) :
- A specification (about 5 pages) describing all the above points
- A Github repo with folder and subfolder structuring adapted to the final solution (even if the files are empty)
- An identified and usable database (accessible from Today I Learned for programmers )
- A trained and evaluated model
- A first version of the API using the model (the commands needed to use it are presented in Today I Learned for programmers )
- Unit tests integrated into the github Actions testing
- the model in the test and train phase
- the various API endpoints
- the database

## Step 2
[ ] Isolation & CI/CD: Model maintenance and continuous monitoring: 25.08.2024 (06.09.2024)
Once the ML model has been put into production, it is essential to monitor its performance and maintain it. When an ML model is running on real-world data, a drop in performance can be observed. Therefore, the best practice to avoid a drop in model performance is to monitor its performance continuously to decide whether it is necessary to re-train the model. The decision resulting from the monitoring leads to an update of the machine learning model.
Points to cover:
- Containerisation of the API, models and DB
- Evaluation of the model under production conditions
- Monitoring the effectiveness and efficiency of in-service model prediction
- Compare with previously specified success criteria (thresholds)
- Re-train the model if necessary
- Collect new data
- Label new data points
- Repeat tasks from the engineering and model evaluation phases (automatically)
- Continuous model integration, training and deployment
Rendering:
- Docker Compose
- CI/CD deployment and monitoring pipeline
- Perform automated tests before deploying code to production to ensure that the code works correctly.

## Step 3
[ ] Demo + Defence: 09.09.2024
For your demo, your application will need to be :
- Organised and documented on GitHub with the installation procedure.
- Works without bugs.
- The DataScientest exam is conducted remotely and in groups as follows:
- 20-minute presentation including
- Presentation of the context and objectives
- Presentation of the architecture in place (model/BDD/API)
- Demonstration of the use of the API and the various monitoring tools (github/Airflow/MLFlow)
- 10 minutes of questions from the members of the jury
