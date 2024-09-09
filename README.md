# jul24_bmlops_plant_recognition: Plant Recognition MLOps

This project was made during the Data Scientist course of [Datascientest](https://datascientest.com/en) and used the datasets described below:
- [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

## Team

- Alex Tavkhelidze - [GitHub](https://github.com/alexbgg)
- Arif Haidari - [LinkedIn](https://www.linkedin.com/in/arif-haidari/), [GitHub](https://github.com/arifhaidari)
- Luigi Menale - [LinkedIn](https://www.linkedin.com/in/lmenale/), [GitHub](https://github.com/lmenale)

Coordinator Sebastien Sime

## Project Organization

    ├── .github
    │   └── workflows                           <- Contains the template for the Pull Request and the CI
    ├──api_auth
    │  ├──app                                   <- Source code for use in this project.
    │  │  ├──database
    │  │  ├──endpoints
    │  │  ├── main.py
    │  │  ├──schemas
    │  │  └──utils
    │  ├── Dockerfile
    │  ├── README.md
    │  ├── requirements.txt
    │  ├──tests
    │  └── wait-for-it.sh
    ├──api_db
    │  ├──app                                   <- Source code for use in this project.
    │  │  ├──database
    │  │  ├──endpoints
    │  │  ├── main.py
    │  │  ├──schemas
    │  │  └──utils
    │  ├── centralised_db_management.yml
    │  ├── Dockerfile
    │  ├── micro_service_architecture.yml
    │  ├── requirements.txt
    │  └──tests
    ├──api_mlflow
    │  ├── docker-compose.yml
    │  ├── Dockerfile
    │  ├──model                                 <- Source code for use in this project.
    │  ├── pyproject.toml
    │  ├── README.md
    │  ├── retrain.sh
    │  ├──test
    │  └── train.sh
    ├──api_prediction
    │  ├──app                                   <- Source code for use in this project.
    │  │  ├──database
    │  │  ├──endpoints
    │  │  ├── main.py
    │  │  ├──schemas
    │  │  └──utils
    │  ├── Dockerfile
    │  ├── README.md
    │  ├── requirements.txt
    │  └──tests
    ├──api_training
    │  ├──app                                   <- Source code for use in this project.
    │  │  ├──database
    │  │  ├──endpoints
    │  │  ├── main.py
    │  │  ├──schemas
    │  │  └──utils
    │  ├── Dockerfile
    │  ├── requirements.txt
    │  └──tests
    ├── docker-compose.yml
    ├── poetry.lock                             <- Dependencies used by poetry.
    ├── pyproject.toml                          <- Package manager.
    ├── README.md                               <- The top-level README for developers using this project.
    ├──reports                                  <- The reports that you'll make during this project as PDF
    ├── requirements.txt
    └── ROADMAP.md                              <- The summary plan for developers using this project.


## Summary plan
Please refer [here](ROADMAP.md)

## How to build the project

Command to build the images
```bash
docker compose build
```

Command to excute the app
```bash
docker compose up
```

We will use this to mount the GDrive folder that contains the MLFlow data and the Dataset.


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
