# 启动一个配置了APOC插件的Neo4j容器实例。这个实例会将当前目录下的 data 和 plugins 文件夹挂载到容器的对应目录，允许你持久化数据和插件，并设置了一些环境变量以启用APOC插件的功能。
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
  neo4j:5.20.0
  
#   启动一个配置了APOC插件的Neo4j容器实例。这个实例会将当前目录下的 data 和  plugins 文件夹挂载到容器的对应目录，允许你持久化数据和插件，并设置了一些环境变量以启用APOC插件的功能。
