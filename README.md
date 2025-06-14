<h1>Project description<h1>


Bachelor Thesis: Lane Detection on microcontrollers: Raspberry Pi Pico 0

Goal of the project is to create lane finding solutions that can run on limited hardware.
That is why the Raspberry Pi Pico 0/2 were selected. The constraints therefore are 264KB/512KB of RAM.
Disclaimer this project focuses on POV detection and does not tackle unwarping of results, 


You can also find tinyML solutions under the same premise as this project at:
- Aykut projet - traffic sign recognition [TODO: add link]
- Alex project - depth recognition [TODO: add link]

_No students were harmed in the completion of this project_	


<h1>Structure<h1>

The code is divided between the Machine Learning approach and a Image processing Based approach

You can find the ML approach and details in /ML
You can find the IP approach and details in /IP
Evaluation code can be found in /Evaluation

License forbids me to share the datasets of BDD100K and TUSIMPLE so here are the download links:
The preprocessing code can be found in the /Dataset folder and will be accompanied with empty folders for each dataset

I apologize as a lot of the references are to be hardwired but I will give a simple guide as to how to link them

Additional Information on how to set up each workflow and running instructions can be found in the respective ReadME's for every section


<h1>Results<h1>

I'm very proud to announce I was able to fit these models on the pico:
- U-net 40*40
- 
- Hough transform based code on the 




