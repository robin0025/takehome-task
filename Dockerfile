# install the latest miniconda (using regularly updated)
FROM continuumio/miniconda3

# Set the working directory inside the container
WORKDIR /app

# Copy environment.yml to a temp location so we update the environment. 
COPY environment.yml .

# Create conda environment based on yml
RUN conda env create -f environment.yml && conda clean --all -f --yes

# Copy source
COPY src ./src
COPY data ./data
COPY result ./result

# initiate venv
RUN echo "source activate takehome-task" > ~/.bashrc
ENV PATH=/opt/conda/envs/takehome-task/bin:$PATH

# move/set run code 
COPY run.py .
ENTRYPOINT ["python", "run.py"]