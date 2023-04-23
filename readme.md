<div style="position: absolute; top: 0; right: 0;">
    <a href="ertugrulbusiness@gmail.com"><img src="https://ssl.gstatic.com/ui/v1/icons/mail/rfr/gmail.ico" height="30"></a>
    <a href="https://tr.linkedin.com/in/ertu%C4%9Fruldemir?original_referer=https%3A%2F%2Fwww.google.com%2F"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" height="30"></a>
    <a href="https://github.com/ertugruldmr"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" height="30"></a>
    <a href="https://www.kaggle.com/erturuldemir"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/kaggle/kaggle-original.svg" height="30"></a>
    <a href="https://huggingface.co/ErtugrulDemir"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" height="30"></a>
    <a href="https://stackoverflow.com/users/21569249/ertu%c4%9frul-demir?tab=profile"><img src="https://upload.wikimedia.org/wikipedia/commons/e/ef/Stack_Overflow_icon.svg" height="30"></a>
    <a href="https://medium.com/@ertugrulbusiness"><img src="https://upload.wikimedia.org/wikipedia/commons/a/a5/Medium_icon.svg" height="30"></a>
    <a href="https://www.youtube.com/channel/UCB0_UTu-zbIsoRBHgpsrlsA"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/YouTube_full-color_icon_%282017%29.svg/1024px-YouTube_full-color_icon_%282017%29.svg.png" height="30"></a>
</div>

# Pet Image Segmentation
 
## __Table Of Content__
- (A) [__Brief__](#brief)
  - [__Project__](#project)
  - [__Data__](#data)
  - [__Demo__](#demo) -> [Live Demo](https://ertugruldemir-animefacegeneration.hf.space)
  - [__Study__](#problemgoal-and-solving-approach) -> [Colab](https://colab.research.google.com/drive/1m07bSVchcD4p4rBUM_1_OQgG6E91S3Pj)
  - [__Results__](#results)
- (B) [__Detailed__](#Details)
  - [__Abstract__](#abstract)
  - [__Explanation of the study__](#explanation-of-the-study)
    - [__(A) Dependencies__](#a-dependencies)
    - [__(B) Dataset__](#b-dataset)
    - [__(C) Modelling__](#c-modelling)
    - [__(D) Deployment as web demo app__](#g-deployment-as-web-demo-app)
  - [__Licance__](#license)
  - [__Connection Links__](#connection-links)
  - NOTE: The model file exceeded limitations. you can download it from this [link](https://huggingface.co/spaces/ErtugrulDemir/AnimeFaceGeneration/resolve/main/GANModel_Weights.zip).

## __Brief__ 

### __Project__ 
- This is an __image generation__  project as  __GAN__ that uses the  [__anime_faces__](https://www.kaggle.com/datasets/soumikrakshit/anime-faces) to __generate anime face__ as a new image.
- The __goal__ is build a deep learning generative adversarial network model that accurately __generates anime face__ from images.
- The performance of the model is evaluated using several __metrics__ loss and accuracy metrics.

#### __Overview__
- This project involves building a deep learning model to generate anime face images. The dataset contains 21551 anime face images.  The models selected according to model tuning results, the progress optimized respectively the previous tune results. The project uses Python and several popular libraries such as Pandas, NumPy, tensorflow.

#### __Demo__

<div align="left">
  <table>
    <tr>
    <td>
        <a target="_blank" href="https://ertugruldemir-animefacegeneration.hf.space" height="30"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" height="30">[Demo app] HF Space</a>
      </td>
      <td>
        <a target="_blank" href="https://colab.research.google.com/drive/1N7eHtdvBjTb9QHk4djcR-EqJJULFolR2"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">[Demo app] Run in Colab</a>
      </td>
      <td>
        <a target="_blank" href="https://github.com/ertugruldmr/AnimeFaceGeneration/blob/main/study.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png">[Traning pipeline] source on GitHub</a>
      </td>
    <td>
        <a target="_blank" href="https://colab.research.google.com/drive/1m07bSVchcD4p4rBUM_1_OQgG6E91S3Pj"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">[Traning pipeline] Run in Colab</a>
      </td>
    </tr>
  </table>
</div>


- Description
    -  __generate anime face image__
    - __Usage__: Set the random seed to generation.
- Embedded [Demo](https://ertugruldemir-animefacegeneration.hf.space) window from HuggingFace Space
    

<iframe
	src="https://ertugruldemir-animefacegeneration.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>

#### __Data__
- The [__anime_faces__](https://www.kaggle.com/datasets/soumikrakshit/anime-faces)  from tensorflow dataset api.
- The dataset contains 21551 anime face images
  - Example Dataset
      <div style="text-align: center;">
        <img src="docs/images/example_data.png" style="width: 400px; height: 300px;">
      </div>
  - Augmented images
      <div style="text-align: center;">
        <img src="docs/images/augment_images.png" style="width: 400px; height: 300px;">
      </div>


#### Problem, Goal and Solving approach
- This is an __generative adversarial network__ problem  that uses the  [__anime_faces__](https://www.kaggle.com/datasets/soumikrakshit/anime-faces)  to __generate anime face images__ from given random seed to create.
- The __goal__ is build a deep learning image classification model that accurately __segments the pets__ from images.
- __Solving approach__ is that using the supervised deep learning models. Basic Custom convolutional modeles are used for generative model and descriminator model. 

#### Study
The project aimed generating anime face images using deep learning model architecture. The study includes following chapters.
- __(A) Dependencies__: Installations and imports of the libraries.
- __(B) Dataset__: Downloading and loading the dataset. Preparing the dataset via tensorflow dataset api. Configurating the dataset performance and related pre-processes. 
- __(C) Preprocessing__: Type casting, value range scaling, resizing, configurating the dataset object, batching, performance setting, visualizating, Implementing augmentation methods on train dataset and image classification related processes.
- __(D) Modelling__:
  - Model Architecture
    - Custom convolutional neural network used sa generator and another custom convolutional network used for discriminator model.
    - Generative Architecture
      <div style="text-align: center;">
        <img src="docs/images/generator_arhitecture.png" style="width: 400px; height: 800px;">
      </div>
    - Disciminator Arhitecture 
      <div style="text-align: center;">
        <img src="docs/images/discriminator_arhitecture.png" style="width: 400px; height: 800px;">
      </div>
  - Training
    - Callbakcs and trainin params are setted. some of the callbacks are EarlyStopping, ModelCheckpoint, Tensorboard etc....    
  - Saving the model
    - Saved the model as tensorflow saved model format.
- __(E) Deployment as web demo app__: Creating Gradio Web app to Demostrate the project.Then Serving the demo via huggingface as live.

#### results
- The final model is __Custom Classifier Network__ because of the results and less complexity.
  -  Custom Generative Model Results
        <table><tr><th>Generation Results </th><th></th></tr><tr><td>
    | model                         | generative loss   | discriminator loss |
    |-------------------------------|--------|----------|
    | Custom Generative Model | 0.6989 | 0.6949 |
    </td></tr></table>

## Details

### Abstract
- [__anime_faces__](https://www.kaggle.com/datasets/soumikrakshit/anime-faces) is used to generate anime face images.  The dataset contains 21551 anime face images. The problem is a generative adversarial task. The goal is generating anime face images using through custom deep learning algorithms or related training approachs of pretrained state of art models. A custom convolutional neural network architecture is used for generative model and the same way a custom convolutional network is used for discriminator model .The study includes creating the environment, getting the data, preprocessing the data, exploring the data, agumenting the data, modelling the data, saving the results, deployment as demo app. Training phase of the models implemented through tensorflow callbacks. After the custom model traininigs, transfer learning and fine tuning approaches are implemented. Selected the basic and more succesful when comparet between other models  is  custom convolutional generative model.__Custom Generative Model__  has __0.6989__ generative loss , __0.6949__ discriminator loss,  other metrics are also found the results section. Created a demo at the demo app section and served on huggingface space.  


### File Structures

- File Structure Tree
```bash
├── demo_app
│   ├── app.py
│   ├── GANModel_Weights
│   └── requirements.txt
├── docs
│   └── images
├── env
│   ├── env_installation.md
│   └── requirements.txt
├── readme.md
└── study.ipynb
```
- Description of the files
  - demo_app/
    - Includes the demo web app files, it has the all the requirements in the folder so it can serve on anywhere.
  - demo_app/GANModel_Weights:
    - Custom Convolutional Model Which saved as tensorflow saved_model format.
  - demo_app/requirements.txt
    - It includes the dependencies of the demo_app.
  - docs/
    - Includes the documents about results and presentations
  - env/
    - It includes the training environmet related files. these are required when you run the study.ipynb file.
  - LICENSE.txt
    - It is the pure apache 2.0 licence. It isn't edited.
  - readme.md
    - It includes all the explanations about the project
  - study.ipynb
    - It is all the studies about solving the problem which reason of the dataset existance.    

### Explanation of the Study
#### __(A) Dependencies__:
  - The libraries which already installed on the environment are enough. You can create an environment via env/requirements.txt. Create a virtual environment then use hte following code. It is enough to satisfy the requirements for runing the study.ipynb which training pipeline.
  - Dataset can download from tensoflow.
#### __(B) Dataset__: 
  - Downloading the [__anime_faces__](https://www.kaggle.com/datasets/soumikrakshit/anime-faces)   via tensorflow dataset api. 
  - The dataset contains 21551 anime face images.
  - Preparing the dataset via resizing, scaling into 0-1 value range, implementing data augmentation and etc image preprocessing processes. 
  - Creating the tensorflow dataset object then configurating.
  - Example Images
    - Example Pure Samples
      <div style="text-align: center;">
        <img src="docs/images/example_data.png" style="width: 400px; height: 300px;">
      </div>
    - Augmented images
      <div style="text-align: center;">
        <img src="docs/images/augment_images.png" style="width: 400px; height: 300px;">
      </div>   
#### __(C) Modelling__: 
  - The processes are below:
    - Archirecture
      - Custom Generative Model
        - Generative Architecture
          <div style="text-align: center;">
            <img src="docs/images/generator_arhitecture.png" style="width: 400px; height: 800px;">
          </div>
        - Disciminator Arhitecture 
          <div style="text-align: center;">
            <img src="docs/images/discriminator_arhitecture.png" style="width: 400px; height: 800px;">
          </div>
      - Training Insights Epoch By Epoch
          <div style="text-align: center;">
            <img src="docs/images/TrainingInsights.gif" style="width: 800px; height: 600px;">
          </div>
    - Results
      -  Custom Generative Model Results
            <table><tr><th>Generation Results </th><th></th></tr><tr><td>
        | model                         | generative loss   | discriminator loss |
        |-------------------------------|--------|----------|
        | Custom Generative Model | 0.6989 | 0.6949 |
        </td></tr></table>
  - Saving the project and demo studies.
    - trained model __fine_tuned_VGG16.h5__ as tensorflow (keras) saved_model format.

#### __(D) Deployment as web demo app__: 
  - Creating Gradio Web app to Demostrate the project.Then Serving the demo via huggingface as live.
  - Desciption
    - Project goal is generating anime face images from given a random number as seed.
    - Usage: upload or select the image for generating then use the button to predict.
  - Demo
    - The demo app in the demo_app folder as an individual project. All the requirements and dependencies are in there. You can run it anywhere if you install the requirements.txt.
    - You can find the live demo as huggingface space in this [demo link](https://ertugruldemir-animefacegeneration.hf.space) as full web page or you can also us the [embedded demo widget](#demo)  in this document.  
    
## License
- This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

<h1 style="text-align: center;">Connection Links</h1>

<div style="text-align: center;">
    <a href="ertugrulbusiness@gmail.com"><img src="https://ssl.gstatic.com/ui/v1/icons/mail/rfr/gmail.ico" height="30"></a>
    <a href="https://tr.linkedin.com/in/ertu%C4%9Fruldemir?original_referer=https%3A%2F%2Fwww.google.com%2F"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" height="30"></a>
    <a href="https://github.com/ertugruldmr"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" height="30"></a>
    <a href="https://www.kaggle.com/erturuldemir"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/kaggle/kaggle-original.svg" height="30"></a>
    <a href="https://huggingface.co/ErtugrulDemir"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" height="30"></a>
    <a href="https://stackoverflow.com/users/21569249/ertu%c4%9frul-demir?tab=profile"><img src="https://upload.wikimedia.org/wikipedia/commons/e/ef/Stack_Overflow_icon.svg" height="30"></a>
    <a href="https://www.hackerrank.com/ertugrulbusiness"><img src="https://hrcdn.net/fcore/assets/work/header/hackerrank_logo-21e2867566.svg" height="30"></a>
    <a href="https://app.patika.dev/ertugruldmr"><img src="https://app.patika.dev/staticFiles/newPatikaLogo.svg" height="30"></a>
    <a href="https://medium.com/@ertugrulbusiness"><img src="https://upload.wikimedia.org/wikipedia/commons/a/a5/Medium_icon.svg" height="30"></a>
    <a href="https://www.youtube.com/channel/UCB0_UTu-zbIsoRBHgpsrlsA"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/YouTube_full-color_icon_%282017%29.svg/1024px-YouTube_full-color_icon_%282017%29.svg.png" height="30"></a>
</div>

