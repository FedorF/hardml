start_runner:
	docker run -d --name gitlab-runner --restart always -v /srv/gitlab-runner/config:/etc/gitlab-runner \
    -v /var/run/docker.sock:/var/run/docker.sock gitlab/gitlab-runner:latest

start_regester:
	docker run --rm -it -v /srv/gitlab-runner/config:/etc/gitlab-runner gitlab/gitlab-runner register

get_secret_number:
	docker run -d --rm -p 5000:5000 waryak/karpov_courses:secret_ci_service
	curl http://localhost:5000/get_secret_number/test
