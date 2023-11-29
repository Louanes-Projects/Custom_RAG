## Introduction
Welcome to our custom RAG (Retrieval Augmented Generation) project! We've crafted this using a micro-services architecture to bring you a powerful, yet flexible, solution.

# What's Inside?
Our project is composed of three key services 🛠️ 

1. Embedding API
This service turns your text into numerical vectors, crucial for our machine learning magic! You have the flexibility to choose between:

    - Sentence-transformers embedding models from Hugging Face.
    - OpenAI Embeddings for advanced performance.

2. Knowledge Retriever
Think of this as the brain of our project. It can:

    - Extracts text from various documents (PDF, Word, PPTX) using the versatile unstructured Python library (also capable of OCR).
    - Manages a 'vectorstore' for efficient similarity searches, powered by ChromaDB.
    - Connects to Azure OpenAI GPT and Vertex AI Palm Text Generation Models, bringing cutting-edge AI into play.

3. Front App
A user-friendly chat interface built with Gradio, making it easy for you to interact with our service.

# Getting Started 

Here's how to get everything up and running 🚀

1. Installation Process
    Simply build and run the images using our Docker compose file.
    Execute this command from the project's root directory: docker compose up --build

2. Software Dependencies
    Our Embedding & Knowledge Retriever services use a PyTorch image from DockerHub.
    For local testing, ensure that the torch library is installed in your Python environment.
    
3. API References
    To extend and enrich our services, we're leveraging:
        - GCP Vertex AI services
        - Azure OpenAI Services


# Deployment

The application is Kubernetes-ready! 🌟

1. Kubernetes Deployment
You'll find everything you need in the Kubernetes Folder within the project.
This includes all the necessary manifests to smoothly deploy the application on a Kubernetes cluster.

2. Continuous Delivery Made Easy

In our Pipelines Folder, we have set up a streamlined continuous delivery process using Azure DevOps Pipelines. There's a dedicated pipeline for each of our three services, and each pipeline includes two key stages:

    - Build and Push: This stage takes care of building each service's image and pushing it to a Container Registry. It ensures that your latest code is always ready to be deployed.

    - Update and Sync: In this stage, the Kubernetes manifests are updated with the commitID. This is crucial for ArgoCD, as it allows for synchronization with the latest image pushed to the container registry.