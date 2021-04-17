Please follow below instructions to run od_app in a containarised environment.
Please make sure that "docker" and "docker-compose" are installed on your machine beforehand.

1. Extract content of the fil1.zip archive.
2. Open terminal and navigate to the directory with extracted content.
3. Execute: "docker build -t od_app ." and wait for Docker image to be created.
4. Execute: "docker-compose -f compose.yaml up" to run both web app and tensorflow prediction 
   service in separate containers and create network mappings between them.
5. Go to one of the following web addresses in your web browser:
   - localhost:5000
   - 0.0.0.0:5000
6. Enjoy your local image prediction service.

In order to close the application:

7. Open new terminal and navigate to the directory with extracted content.
8. Execute: "docker-compose -f compose.yaml down" to close both containers.

Known issues:
- Bird detection model was trained and inferenced using jupyter notebook. The results of inference were better
  than those on the same images when running web app. Currently this issue is unresolved and will require further
  testing to be performed.
- Please use pictures of 1000px width and heigh (+- 300px). Pictures smaller than 700px are too small and objects
  will likely not be detected. While objects on pictures with higher resolutions are usually being detected it
  might be difficult to spot detection boxes in some cases. Further development will include dynamically changing
  box line width depending on picture size.
