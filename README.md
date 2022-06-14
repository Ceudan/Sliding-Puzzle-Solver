\documentclass{article}
% if you need to pass options to natbib, use, e.g.:
%     \PassOptionsToPackage{numbers, compress}{natbib}
% before loading neurips_2021

% ready for submission
\usepackage{neurips_2021}

% to compile a preprint version, e.g., for submission to arXiv, add add the
% [preprint] option:
%     \usepackage[preprint]{neurips_2021}

% to compile a camera-ready version, add the [final] option, e.g.:
%     \usepackage[final]{neurips_2021}

% to avoid loading the natbib package, add option nonatbib:
%    \usepackage[nonatbib]{neurips_2021}

\usepackage[utf8]{inputenc} % allow utf-8 input
\usepackage[T1]{fontenc}    % use 8-bit T1 fonts
\usepackage{hyperref}       % hyperlinks
\usepackage{url}            % simple URL typesetting
\usepackage{booktabs}       % professional-quality tables
\usepackage{amsfonts}       % blackboard math symbols
\usepackage{nicefrac}       % compact symbols for 1/2, etc.
\usepackage{microtype}      % microtypography
\usepackage{xcolor}         % colors
\usepackage{graphicx}       % insert images
\usepackage[style=ieee]{biblatex}
\usepackage{svg}            % insert svg
\addbibresource{references.bib}

\title{Solving the Jigsaw Puzzle via Various Learning Approaches}

% The \author macro works with any number of authors. There are two commands
% used to separate the names and addresses of multiple authors: \And and \AND.
%
% Using \And between authors leaves it to LaTeX to determine where to break the
% lines. Using \AND forces a line break at that point. So, if LaTeX puts 3 of 4
% authors names on the first line, and the last on the second line, try using
% \AND instead of \And before the third author name.

\author{%
Kyra Chow \And Ray Coden Mercurius \And Anna Xu \\
  Department of Computer Science\\
  University of Toronto\\
  Toronto, ON M5S2E4 \\
  \texttt{csc413-2022-01@cs.toronto.edu} \\
}

\begin{document}

\maketitle

\begin{abstract}
    The jigsaw puzzle is a game dating back to the 1760s and despite years of
    developments in both game and technology, there is a lack of literature
    utilizing various machine learning (ML) techniques to solve this game. To tackle this issue, we
    developed a learning-based solver which draws inspiration from multiple existing ML
    methods involving computer vision. Through analyses of each model's performance of solving the puzzle, we then determine whether certain model architectures are preferred over others for this game. 
\end{abstract}

\section{Introduction}
      A square jigsaw is a subset of jigsaw puzzles where each puzzle tile is a square and the correct configuration of the tiles displays an n-by-n grid. To solve a square jigsaw, the user must first determine the correct tile configuration (position and orientation). In this project, we focus on using computer vision techniques in machine learning (ie. convolutional neural networks (CNNs)) to identify the correct tile configuration. We will then compare the solving performance of various Deep CNN (DCNN) architectures. The motivation for this project follows from the lack of diverse machine learning-based literature surrounding jigsaw DCNN solvers. We hope that our conclusions may be be leveraged to develop more complex algorithms for other visual puzzles such as non-square jigsaws or sliding puzzles.
  

\section{Related Works} 
    There are 2 general approaches to solving square jigsaws.
    
    Firstly, one can solve the board in an iterative process. This is done by computing an edge compatibility metric between sub-image pairs, and using this information to prune possibilities. Zanoci and Andress computed an edge compatibility metric using non-ML neighboring pixel information and the color gradients \cite{zanoc}. This was done between all possible sub-image pairs ($ n^{2}*16$ total comparisons, where n = number of sub-images). Then, treating pieces as vertices and compatibility scores as edges, they utilized the minimum spanning tree algorithm to recreate the original board using the most likely edges. Only 80\% of edges were correctly matched, using board sizes of hundreds of pieces. A downfall of non-ML methods is the deterministic nature of the solver, and its difficulty to adapt to simple perturbations such as eroded edges. A simpler method proposed by Bhaskhar, uses a fine-tuned ResNet to compute edge compatabilities between puzzle tiles. Then the grid is iteratively filled in, using the most probable tile at each time step. As a result, over 80\% of pieces were rearranged into their correct position/orientation \cite{unpuzzled}.
    
    The second approach is for one to attempt to solve the entire board at one time. Noroozi separately fed each sub-piece into a siamese-ennead CNN, then combined these encodings in a fully connected layer to predict the original permutation \cite{noroozi_favaro_2016}. Accuracy in terms of solving entire board was low at only 67\% with a board size of 3x3. Given that 9! permutations exist, solving at once may be too complex for current architectures. A similar method by Kulharia \cite{kulharia} used a many to one long short-term memory (LSTM) to output a one-hot encoding of the permutation. Input at each time-step was a collapsed sub-image. It only achieved 80\% accuracy per complete board solve despite a small board size of 2x2. We will expand on this approach in our project. 


\section{Method}
    \subsection{Image Data Source}
        We will create the dataset for our report using the Caltech-UCSD Birds 200 (CUB\_200) 2011 dataset \cite{cub200}. CUB-200 2011 is an image dataset containing 11,788 images of 200 different bird species. Using a 70/30 split, we will partition these images into training and validation datasets, with 8,252 images in the training set and 3,536 in the validation set. 

    \subsection{Edge Compatibility Train Data} 
        Our DCNN's input is a square concatenation of 2 subimages, and the output is whether they are left-right adjacent, which we will now refer to as adjacent. To generate these input output pairings  we make use of Bhaskhar's \texttt{Checking\_adjacency\_dataset} repository \cite{unpuzzled}, which will generate edge concatenations between the images' various edges (See Fig \ref{Edge Data}). For each n-by-n grid, there are n$^{2}$ puzzle pieces and C(n$^{2}$, 2) possible pairs of square concatenations. The repository will then randomly choose from these pairs to create our train data that has a roughly equal number of adjacent labels, and non-adjacent labels, and will output our newly formed data. 
        
        \begin{figure}[!ht]
            \centering
            \includegraphics[width = 0.9\textwidth]{data generation.png}
            \caption{2 sub-images can generate 4 different concatenations.}
        \end{figure}
        
\subsection{Adjacency Classifier} 
    The goal of this classifier is to determine whether an input image, consisting of two puzzle pieces, (P,Q), is left-right adjacent. Drawing inspiration from the work of \cite{unpuzzled}, we will finetune different image classification models, such as ResNet, on a custom adjacency dataset to create the adjacency classifier. Furthermore, we want to compare the performance of these models both in terms of their prediction accuracy as adjacency classifiers, and how well they work in combination with the downstream solver to complete the puzzle. In the following sections we will briefly describe the architecture of each CNN used, and how we have adapted the model to fit our finetuning process. 
    
    \subsubsection{ResNet Architecture}
    The ResNet models were first proposed in \cite{ResNet}. There are currently five versions of ResNet (ResNet18, ResNet34, ResNet50, ResNet101, ResNet152) trained on the ImageNet dataset available for fine tuning in the \texttt{torchvision} library in PyTorch. In general, the ResNet models contain convolutional kernels of size 3x3, and the number of filter layers increases along with the depth of the network. Thus, different versions of the ResNets correspond to the total number of filter layers in the network (see Fig. \ref{ResNet architecture} for architecture detail). The defining feature for ResNets are the residual blocks which exist between pairs/triplets of the convolutional layers with residual connections that propagate the unaltered input to the output (see Fig. \ref{ResNet residual block}). This innovation was introduced in order to prevent the vanishing gradient problem. \\

    To adapt ResNets for our finetuning process, we modified the final fully connected layer originally meant for a larger classification task to one meant for binary classification. Note that when we finetuned on all the ResNets, the weights/bias parameters that are affected belong to this final classification layer. 

    \subsubsection{VGG Architecture}
    The VGG models were first proposed in \cite{VGG}. There are 4 versions of VGG (VGG11, VGG13, VGG16, VGG19) with batch normalization implemented in the \texttt{torchvision} library in PyTorch. Similar to ResNets, the VGGs contain only 3x3 convolution filters; however, they are considerably more shallow (the deepest VGG has 19 total layers, while the deepest ResNet has 152) (see. Fig. \ref{VGG architecture}). The limitation on the VGGs' depth is due to the vanishing gradient problem, where more layers lead to issues with weight updates during training.
    
    To adapt the VGGs for our finetuning process, we have to again modify the final classification layer to fit our binary classification task.
    
    \subsubsection{ResNeXt Architecture}
    
    The ResNeXt models were first proposed in \cite{ResNeXt}(See Fig. \ref{ResNeXt architecture}). What sets the ResNeXts apart from  ResNets are the structure of the residual blocks. Instead of applying convolution filters to the block's entire input feature map, a series of convolutions are applied to parts of the the input and then the output is then grouped back together(see Fig. \ref{ResNeXt residual block}). This parallel structure allows specialization to occur among groups where different characteristics are learned (ie. edges, corners etc). \\
    
    To adapt the ResNeXts for our finetuning process, we modified the final fully connected layer exactly like how we had done for the ResNets. 
    
    \subsubsection{Setup}
    For the sake of fair comparison, all CNN models were tested on control hyper parameters (see Table \ref{tab:hyperparameters}).

\subsection{Configuration Solver}

    The configuration solver takes as input a randomly scrambled n-by-n board (see Fig. \ref{OG} - \ref{final_solution}). Due to time constraints, this portion of the implementation leveraged existing functions from \cite{unpuzzled}. At each time step the solver receives a board that is partially filled in, and outputs a new board with 1 additional piece adjacent to a piece already filled in. To decide on the next piece, the solver will try every single combination of (open spot, remaining piece, orientation). That is if there are S open spots, R remaining pieces, and 4 orientations, our solver will attempt S*R*4 combinations, and pick the most probable one according to the average edge compatibility scores of all its edges. The validity of an edge of 2 pieces stacked vertically can be checked by simply cutting out the edge, rotating the edge 90 degrees, and then passing it in to the left-right adjacent CNN classifier. The solver process is visualized in Figure \ref{solver architecture} of the appendix.
    
    \begin{figure}[!ht]
        \centering
        \includegraphics[width=1\textwidth]{Solver_Architecture.png}
        \caption{Architecture of Solving Process}
        \label{solver architecture}
    \end{figure}

    % CODEN PLEASE WRITE ABOUT HOW TOP-BOTTOM IS THE SAME AS LEFT RIGHT
\section{Results and Discussion}

\subsection{Adjacency Classifier Performance} 
Overall all, we see that all models achieved their highest validation accuracy between 0.90 and 0.94, as their lowest validation loss between <0.15 and 0.36. In general, we see that deeper, more complex models do not offer better generalizability for our problem, instead, shallower, simpler models are more capable. 

In terms of individual model performances, we achieved the highest training accuracy of 0.99 and lowest training loss of 0.05 on VGG13, and highest validation accuracy of 0.94 and lowest validation loss of <0.15 on VGG16. 

\subsection{Final Solver}
To test how well our adjacency classifier works in combination with the downstream configuration solver, we run the entire pipeline (ie. adjacency classifier + configuration solver) on two test images (see Fig. \ref{OG}-\ref{final_solution} for sample input/output). In the interest of time we did not run a complete performance study with more samples. However, what we observe from our limited test set is that VGG16 performed the best (ie. it gave a correct complete solution) in terms of solving a 3x3 board, and a 2x2 board. This result corroborate with our above observation as the adjacency classifier created using VGG16 gave the best validation loss/accuracy (see Table. \ref{3x3results} -\ref{2x2results}).  

\subsection{Discussion}

Looking at our training (Fig. \ref{resnet_train}) and validation (Fig. \ref{resnet_valid}) curves, we observe that the bigger ResNet models (ie. ResNet101 and ResNet152) suffers overfitting - we see the training accuracy improving whilst the validation accuracy worsens (the same can be seen for the loss curves). This is reasonable as the size of our entire dataset is rather small which means that more complex models will easy overfit to the training data and perform suboptimally in validation. 

For the VGG curves, however, the validation curves are not smooth and seem to have jumps in accuracy and loss values. This suggest to us that the learning rate for the model is set too high - the gradient descent algorithm has trouble stabilizing near the local minimum because the step size for updates are too large which causes large oscillations. In the interest of time we did not remedy this issue but it can be simple done by training on a lower learning rate.  

Looking at the ResNeXt curves, we see a different trend occurring where ResNeXt101, specifically, seems to be underfitting. There are multiple reasons why the deeper model may be underfitting, such as it was not trained for long enough, or there is not enough data. ResNeXt50 does not appear to underfit in the same manner as ResNeXt101, but further exploration into the models must be done for this trend to be understood better. 

Importantly, all Deep CNNs hit an adjacency classfication accuracy cap near 93\%. We speculate that this issue may be due to the nature of the dataset itself. Our dataset contains images of birds with natural landscapes that include the sky and water. Unlike in commercial puzzles, our pieces were never checked to maintain a good balance of distinguishing features across all sub images. It is possible that many sub images contained the same generic background (Refer to Figure \ref{demarcated edges} in Appendix) lacking in major features. Here, we require our CNN to pay strong attention to pixel level details. This might be challenging to Deep CNNs pretrained on object recognition, where localized pixel level information is lost in its deeper later layers, especially since we only fine tuned the final classification layer.

\begin{figure}[!ht]
    \centering
    \includegraphics[width = 0.4\textwidth]{demarcated edges.jpg}
    \caption{The edges marked in red do not have defining features. This imbalance across puzzle tiles could have yielded the accuracy cap of 95\%.}
    \label{demarcated edges}
\end{figure}

Finally, we want to observe the effect of using different CNNs for adjacency classification has on the downstream solver. We see that in general there is a positive correlation between good adjacency classification performance and good solver performance. However, we see that the overall solver performance can be improved (only VGG16 + solver gave a completely correct solution for both test samples). 

% Second, our learning rate was set too high. The lack of validation over fitting suggests that we could have trained for longer with a gradually decreases rates. However computational power and data constraints prevented us from performing the rigorous training found in most literature. 

\section{Conclusion and Next Steps}
     In this report, we expanded on existing methods by showing that various pretrained Deep CNNs can be successfully fine-tuned to generate working adjacency scoring metrics. Importantly, we concluded that smaller, wider CNNs (such as VGGs) obtain higher accuracies by paying finer attention to details than longer, narrower models (such as ResNets) that are pretrained for object classification.

    \subsection{Future Work - Train Data}
    In real life, jigsaw tile edges are eroded inwards anywhere from 1\% to 5\%, which would be a perfect challenge for machine learning. In future works, we can test and explore the effects of various degrees of erosion on edge adjacency classification performance.
    
    % Okay I think we should be good to submit now
    \subsection{Future Work - Solver}
    As well, in the future we can tackle the issue of the propagation of errors downstream. Incorrectly placed pieces cannot be readjusted, as well some pieces are placed with only 1 neighboring edge information which introduces significant uncertainty. To remedy this, we propose solving in batches. That is, we can simultaneously solve divisions of 2x2 at a time by brute forcing 8!/4! combinations, taking the most likely combination according to chained edge compatibility scores. Finally, we can have a top-down double check by having a CNN take in a subdivision and output its validity, then backtracking if necessary.

\medskip

{
\small

}
\printbibliography

\appendix

\section*{Appendix}
        % FIGURE 1 - in 3.1.2 EDGE COMPATIBILITY TRAIN DATA
        \begin{figure}[!ht]
        \centering
        \includegraphics[width =0.8\textwidth]{Adjacency_data.png}
        % Need to fix caption
       \caption{Two square concatenations of puzzle piece edges. On the left, we see that the edges are not left-right adjacent and thus they will have label 0. Whereas on the right we see that the edges are left-right adjacent, and thus they will have label 1.}
       \label{Edge Data}
       \end{figure}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 3.2 illustration of input/output of the adjacency classifier
    
    % 3.2.1 ResNet architecture summary table
    \begin{figure}[!ht]
    \centering
         \includegraphics[width=0.9\textwidth]{resnet_architecture.png}
       \caption{Model architecture comparison between the different ResNets}
       \label{ResNet architecture}
    \end{figure}
    % 3.2.1 ResNet residual blocks
    \begin{figure}[!ht]
        \centering
         \includegraphics[width=0.5\textwidth]{resnet_residual.png}
       \caption{Illustration of a residual block structure}
        \label{ResNet residual block}
    \end{figure}
    % 3.2.2 VGG architecture summary table
    \begin{figure}[!ht]
        \centering
         \includegraphics[width=0.5\textwidth]{vgg_architecture.png}
       \caption{Model architecture comparison between the different VGGs}
        \label{VGG architecture}
    \end{figure}
    % 3.2.3 ResNeXt architecture summary table - Not right picture
    \begin{figure}[!ht]
        \centering
         \includegraphics[width=0.5\textwidth]{resnext_architecture.png}
       \caption{Model architecture comparison between ResNet50 and ResNeXt50}
        \label{ResNeXt architecture}
    \end{figure}
    % 3.2.3 ResNeXt residual blocks
    \begin{figure}[!ht]
        \centering
         \includegraphics[width=0.5\textwidth]{resnext_residual.png}
       \caption{Illustration of a ResNeXt residual block structure compared to the a ResNet residual block structure}
        \label{ResNeXt residual block}
    \end{figure}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
    % TABLE 1 - in 3.2.4 SETUP 
    \begin{table}[!ht]
        \centering
        \begin{tabular}{||c c||} 
     \hline
     Hyperparameter  & Value\\ [0.5ex] 
     \hline\hline
     learning rate & 0.001 \\ 
     \hline
     momentum & 0.9  \\
     \hline
     examples per epoch & 500 \\
     \hline
     optimizer & optim.SGD \\[1ex]
     \hline
    \end{tabular}
        \caption{Hyperparameters for all CNN Models}
        \label{tab:hyperparameters}
    \end{table}

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
 % TABLE 2 & 3 - IN 4.1.1 RESULTS
        \begin{table}[!ht]
        \centering
        \begin{tabular}{||c c c||} 
         \hline
         Model & Lowest Training Loss & Highest Training Accuracy \\ [0.5ex] 
         \hline\hline
         Resnet18 & 0.10 & 0.97 \\ 
         \hline
         Resnet34 & 0.11 & 0.96  \\
         \hline
         Resnet50 & 0.10 & 0.96 \\
         \hline
         Resnet101 & 0.14 & 0.95 \\
         \hline
         Resnet152 & 0.09 & 0.97 \\
         \hline
         VGG11 & 0.05 & 0.98 \\
         \hline
         VGG13 & \textbf{0.05} & \textbf{0.99} \\
         \hline
         VGG16 & 0.06 & 0.98 \\
         \hline
         VGG19 & 0.06 & 0.98 \\
         \hline
         ResNeXt50 & 0.11 & 0.97 \\
         \hline
         ResNeXt101 & 0.31 & 0.92 \\[1ex] 
         \hline
        \end{tabular}
        \caption{The lowest training loss and highest training accuracy observed for each model}
        \label{tab:train table}
        \end{table}
        
        \begin{table}[!ht]
        \centering
        \begin{tabular}{||c c c||} 
         \hline
         Model & Lowest Validation Loss & Highest Validation Accuracy\\ [0.5ex] 
         \hline\hline
         Resnet18 & 0.31 & 0.92 \\ 
         \hline
         Resnet34& 0.27 & 0.93  \\
         \hline
         Resnet50 & 0.35 & 0.92 \\
         \hline
         Resnet101 & 0.29 & 0.93 \\
         \hline
         Resnet152 & 0.36 & 0.93 \\
         \hline
         VGG11 & 0.23 & 0.92 \\
         \hline
         VGG13 & 0.27 & 0.92 \\
         \hline
         VGG16 & \textbf{<0.15} & \textbf{0.94} \\
         \hline
         VGG19 & 0.20 & 0.91 \\
         \hline
         ResNeXt50 & 0.26 & 0.92 \\
         \hline
         ResNeXt101 & 0.21 & 0.92 \\[1ex] 
         \hline
        \end{tabular}
        \caption{The lowest validation loss and highest validation accuracy observed for each model}
        \label{tab:val table}
        \end{table}
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
   % SVG GRAPHS
   
   \begin{table}[!ht]
        \centering
        \begin{tabular}{||c c||} 
         \hline
         Model  & Colour\\ [0.5ex] 
         \hline\hline
         Resnet18 & Orange \\ 
         \hline
         Resnet34 & Dark Blue  \\
         \hline
         Resnet50 & Red \\
         \hline
         Resnet101 & Light Blue \\
         \hline
         Resnet152 & Pink \\[1ex]
         \hline
        \end{tabular}
        \caption{Colour Code for Resnet Graphs}
        \label{tab:resnet colours}
    \end{table}
   
        \begin{figure}[!ht]
        \centering
        \includesvg{resnet_Training_Accuracy_Average.svg}
        \caption{Training Accuracy for Resnet Models}
        \label{resnet_train}
        \end{figure}
        
        \begin{figure}[!ht]
        \centering
        \includesvg{resnet_Validation_Accuracy_Average.svg}
        \caption{Validation Accuracy for Resnet Models}
        \label{resnet_valid}
        \end{figure}
        
        \begin{figure}[!ht]
        \centering
        \includesvg{resnet_Training_Loss_Average.svg}
        \caption{Training Loss for Resnet Models}
        \label{resnet_train_loss}
        \end{figure}
        
        \begin{figure}[!ht]
        \centering
        \includesvg{resnet_Validation_Loss_Average.svg}
        \caption{Validation Loss for Resnet Models}
        \label{resnet_valid_loss}
        \end{figure}
        
    \begin{table}[!ht]
        \centering
        \begin{tabular}{||c c||} 
         \hline
         Model  & Colour\\ [0.5ex] 
         \hline\hline
         VGG11 & Orange \\ 
         \hline
         VGG13 & Dark Blue  \\
         \hline
         VGG16 & Red \\
         \hline
         VGG19 & Light Blue \\
         \hline
         ResNeXt50 & Pink \\
         \hline
         ResNeXt101 & Green \\[1ex]
         \hline
        \end{tabular}
        \caption{Colour Code for VGG and ResNeXt Graphs}
        \label{tab:VGG colours}
    \end{table}
        
        \begin{figure}[!tbp]
        \centering
        \includesvg{VGG_Training_Accuracy_Average.svg}
        \caption{Training Accuracy for VGG and ResNeXt Models}
        \label{vgg_train}
        \end{figure}

        \begin{figure}[!tbp]
        \centering
        \includesvg{VGG_Validation_Accuracy_Average.svg}
        \caption{Validation Accuracy for VGG and ResNeXt Models}
        \label{vgg_valid}
        \end{figure}
        
        \begin{figure}[!ht]
        \centering
        \includesvg{VGG_Training_Loss_Average.svg}
        \caption{Training Loss for VGG and ResNeXt Models}
        \label{vgg_train_loss}
        \end{figure}
        
        \begin{figure}[!ht]
        \centering
        \includesvg{VGG_Validation_Loss_Average.svg}
        \caption{Validation Loss for VGG and ResNeXt Models}
        \label{vgg_valid_loss}
        \end{figure}
        
        
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
 
    
% In 4.2 Final solver

% sample input test image
\begin{figure}[!ht]
    \centering
    \includegraphics[width=0.5\textwidth]{OG.png}
      \caption{Original test image}
      \label{OG}
\end{figure}
\begin{figure}[!ht]
    \centering
    \includegraphics[width=0.5\textwidth]{input_to_solver.png}
      \caption{Sample input to solver}
      \label{input_to_solver}
\end{figure}
\begin{figure}[!ht]
    \centering
    \includegraphics[width=0.5\textwidth]{final_solution.png}
      \caption{Final solver output}
      \label{final_solution}
\end{figure}

% charts comparing correct rotation/position
\begin{table}[!ht]
        \centering
        \begin{tabular}{|c|c|c|} 
         \hline
         Model Name  & \# of correct pos. & \# of correct pos. and rot. \\ [0.5ex] 
         \hline
         ResNet18 & 3 & 3  \\
          \hline
         ResNet34 & 1 & 1\\
          \hline
         ResNet50 & 1 &  1 \\
          \hline
         ResNet101 & 3 & 3 \\
          \hline
         ResNet152 & 1 & 1 \\
          \hline
         \textbf{VGG11} & \textbf{9} & \textbf{9} \\ 
         \hline
         VGG13 & 4 & 4 \\
         \hline
         VGG16 & 1 & 1 \\
         \hline
         VGG19 & 2 & 1\\
         \hline
         ResNeXt50 & 6 & 6 \\
         \hline
         ResNeXt101 & 3 & 3\\[1ex]
         \hline
        \end{tabular}
        \caption{Solver result for a 3x3 board. The correct board should feature 9 correct positions and 9 correct rotations}
        \label{3x3results}
    \end{table}
    
\begin{table}[!ht]
        \centering
        \begin{tabular}{|c|c|c|} 
         \hline
         Model Name  & \# of correct pos. & \# of correct pos. and rot. \\ [0.5ex] 
         \hline
         \textbf{ResNet18} & \textbf{4} & \textbf{4}  \\
          \hline
         ResNet34 & 2 & 1 \\
          \hline
         ResNet50 & 2 & 2 \\
          \hline
         \textbf{ResNet101} & \textbf{4} & \textbf{4}  \\
          \hline
         ResNet152 & 4 & 2 \\
          \hline
         \textbf{VGG11} & \textbf{4} &\textbf{4} \\ 
         \hline
         \textbf{VGG13} & \textbf{4} & \textbf{4} \\
         \hline
         VGG16 & 2 & 2 \\
         \hline
         \textbf{VGG19} & \textbf{4} &\textbf{4} \\
         \hline
        \textbf{ ResNeXt50} & \textbf{4} & \textbf{4} \\
         \hline
        \textbf{ ResNeXt101} & \textbf{4} & \textbf{4}\\[1ex]
         \hline
        \end{tabular}
        \caption{Solver result for a 2x2 board. The correct board should feature 4 correct positions and 4 correct rotations}
        \label{2x2results}
    \end{table}
\end{document}
