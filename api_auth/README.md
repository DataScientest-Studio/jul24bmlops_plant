api_auth/
├── Dockerfile
├── app/
│ ├── **init**.py
│ ├── main.py
│ ├── config.py
│ ├── utils/
│ │ ├── **init**.py
│ │ └── auth_utils.py
│ ├── endpoints/
│ │ ├── **init**.py
│ │ ├── api_request_log.py
│ │ ├── error_log.py
│ │ └── authentication.py
│ ├── schemas/
│ │ ├── **init**.py
│ │ ├── api_request_log_schema.py
│ │ ├── error_log_schema.py
│ │ └── auth_schema.py
│ └── database/
│ ├── **init**.py
│ └── db.py
│ └── tables.py
└── requirements.txt
