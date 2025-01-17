# Adaptive Diffusion of Sensitive Information in Online Social Networks

## Overview
This project analyzes and controls the spread of sensitive information in online social networks. It is developed using **Python (Django)** for the backend, **SQL** for data storage, and **HTML and CSS** for the frontend.

### Key Features
- **Dynamic Diffusion Control**: Allows only non-sensitive information to diffuse, while controlling the spread of sensitive data.
- **Round-by-Round Diffusion Analysis**: Tracks diffusion patterns over time.
- **Data Visualization (Optional)**: Shows trends and effectiveness of control mechanisms.

---

## Getting Started

### Prerequisites
1. **Python 3.7+**
2. **Django 3.2+**
3. **SQL Database** (e.g., PostgreSQL, MySQL, SQLite)
4. **pip** (Python package manager)

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/yourprojectname.git
cd yourprojectname
```

### 2. Set Up a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Create the Database

- **For SQLite** (default): No additional setup is needed; SQLite will create a `db.sqlite3` file in the project directory.
- **For PostgreSQL or MySQL**:
  1. Install the database.
  2. Create a new database (e.g., `adaptive_diffusion_db`).
  3. Create a new user and assign privileges.

### 4. Configure `settings.py`

Open `settings.py` and adjust the database settings according to your database of choice.

```python
# settings.py

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',  # For PostgreSQL
        'NAME': 'adaptive_diffusion_db',           # Your database name
        'USER': 'your_db_user',                    # Your database username
        'PASSWORD': 'your_db_password',            # Your database password
        'HOST': 'localhost',                       # Database server address (e.g., localhost)
        'PORT': '5432',                            # Database port (default for PostgreSQL)
    }
}
```

*For other databases, change the `'ENGINE'` key to the appropriate backend, like `django.db.backends.mysql` for MySQL or `django.db.backends.sqlite3` for SQLite.*

---

## Database Setup

1. **Apply Migrations** to create the database schema:
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

---

## Running the Project

1. **Start the Django Development Server**:
   ```bash
   python manage.py runserver
   ```

2. **Access the Application**:
   Open your web browser and go to [http://127.0.0.1:8000/](http://127.0.0.1:8000/) to access the main application.

3. **Admin Panel**:
   For admin management, visit [http://127.0.0.1:8000/admin/](http://127.0.0.1:8000/admin/).

---

## Project Structure

- **app_name**: Contains core application logic (views, models, and templates).
- **static**: Stores static files like CSS, JavaScript, and images.
- **templates**: HTML templates for the frontend.
- **media**: (Optional) Stores user-uploaded content, if any.
- **db.sqlite3**: SQLite database file (only if SQLite is used).

---

## Key Commands

- **Run Development Server**:
  ```bash
  python manage.py runserver
  ```

- **Make Migrations**:
  ```bash
  python manage.py makemigrations
  ```

- **Apply Migrations**:
  ```bash
  python manage.py migrate
  ```

---

## Contributing
Pull requests are welcome. For significant changes, please open an issue first to discuss what you would like to change.


