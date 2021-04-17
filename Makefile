DOCKER_REGISTRY   ?= hub.docker.com
DOCKER_REPO       ?= darshika/mrcnn
BUILD_TAG         := latest
TS                := $(shell date "+%Y%m%d%H%M%S")
NAME              := $(lastword $(subst /, ,$(DOCKER_REPO)))

CI_REPO_URL       ?= $(shell git remote get-url origin)
CI_COMMIT         ?= $(shell git rev-parse --short HEAD)

# Use cached builds on macOS
UNAME_S := $(shell uname -s)
ifneq ($(UNAME_S),Darwin)
	DOCKER_BUILD_ARGS ?= "--no-cache"
endif

IMAGE_ID_FILE     := .iidfile
IMAGE_ID          ?= $(shell cat $(IMAGE_ID_FILE))
RELEASE_TAG       := $(shell cat VERSION)-$(TS)-$(CI_COMMIT)
RELEASE_IMAGE     := $(DOCKER_REGISTRY)/$(DOCKER_REPO):$(RELEASE_TAG)
RELEASE_IMAGE_NO_COMMISH := $(DOCKER_REGISTRY)/$(DOCKER_REPO):$(BUILD_TAG)

all: clean build tag release
local: clean build tag

clean:
	-docker stop $(NAME)
	-docker rm $(NAME)
	-docker rmi -f $(shell cat  $(IMAGE_ID_FILE)) 2>/dev/null
	-rm -f .iidfile

list:	
	@echo $(RELEASE_IMAGE)
	@echo $(RELEASE_IMAGE_NO_COMMISH)
	@echo $(NAME)

build:
	docker build $(DOCKER_BUILD_ARGS)\
		--label org.label-schema.name=$(DOCKER_REPO) \
		--label org.label-schema.version=$(NAME)-$(shell cat VERSION) \
		--label org.label-schema.schema-version=1.0.0-rc.1 \
		--label org.label-schema.vcs-ref=$(CI_COMMIT) \
		--label org.label-schema.vcs-url=$(CI_REPO_URL) \
		--iidfile=$(IMAGE_ID_FILE) \
		.

build-local:
	docker build -t $(NAME) .

tag: 
	docker tag $(IMAGE_ID) $(RELEASE_IMAGE)
	docker tag $(IMAGE_ID) $(RELEASE_IMAGE_NO_COMMISH)

release: 
	docker tag $(IMAGE_ID) $(RELEASE_IMAGE)
	docker tag $(IMAGE_ID) $(RELEASE_IMAGE_NO_COMMISH)
	docker push $(RELEASE_IMAGE)

run-local:
	@./run-local.sh run $(IMAGE_ID) 

run-local-prebuilt:
	@./run-local.sh run  $(RELEASE_IMAGE_NO_COMMISH) 

run-local-clean:
	@./run-local.sh destroy