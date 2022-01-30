init: build start

start:
	@docker-compose up -d

stop:
	@docker-compose stop

restart: stop start

build:
	@docker-compose build

clean:
	@docker-compose down

pull:
	@git checkout .
	@git pull