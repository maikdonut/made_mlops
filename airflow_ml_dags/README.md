### HW3
### To run
```
export FERNET_KEY=$(python -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")
docker-compose up --build
```
### To stop
```
docker-compose down
```