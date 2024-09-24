# 启动一个配置了APOC插件的Neo4j容器实例。这个实例会将当前目录下的 data 和 plugins 文件夹挂载到容器的对应目录，允许你持久化数据和插件，并设置了一些环境变量以启用APOC插件的功能。
#  This instance is ServerId{6e48f114} (6e48f114-de19-4c1b-a08b-ea7ecef64c51)  80bb2e3ff916
# sudo docker restart 37a5fd711f5e 
# sudo docker start start neo4j-v5-apoc 
# sudo docker exec -it  37a5fd711f5e /bin/bash  # 进入容器修改一些东西~
sudo docker start neo4j-v5-apoc
docker run \
  -p 7474:7474 -p 7687:7687 \
  -v ${PWD}/data:/data \
  -v ${PWD}/plugins:/plugins \
  --name neo4j-v5-apoc \
  -e NEO4J_apoc_export_file_enabled=true \
  -e NEO4J_apoc_import_file_enabled=true \
  -e NEO4J_apoc_import_file_use_neo4j_config=true \
  -e NEO4J_PLUGINS='["apoc"]' \
  -e NEO4J_dbms_security_procedures_unrestricted="apoc.*" \
  -e NEO4J_dbms_security_auth_enabled=false \
  neo4j:5.20.0
  
#   启动一个配置了APOC插件的Neo4j容器实例。这个实例会将当前目录下的 data 和  plugins 文件夹挂载到容器的对应目录，允许你持久化数据和插件，并设置了一些环境变量以启用APOC插件的功能。
