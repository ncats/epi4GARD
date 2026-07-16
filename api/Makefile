REPO	:= sci-docker.ncats.nih.gov
NAME	:= ${REPO}/epi-api
TAG		:= $$(git log -1 --pretty=%h)
IMG		:= ${NAME}:${TAG}
LATEST 	:= ${NAME}:latest

build:
	@docker build -t ${IMG} .
	@docker tag ${IMG} ${LATEST}

push:
	@docker push --all-tags ${NAME}

login:
	@docker login ${REPO} -u ${DOCKER_USER} -p ${DOCKER_PASS}
