build_image:
	docker build -t localhost:5000/wep_app .

push_image:
	docker push localhost:5000/wep_app

pull_image:
	docker pull 135.181.204.59:5000/wep_app

init_swarm:
	docker swarm init

run_service:
	docker service create -p 2000:2000 --replicas 6 --name web_app 135.181.204.59:5000/web_app

stop_service:
	docker service rm web_app
