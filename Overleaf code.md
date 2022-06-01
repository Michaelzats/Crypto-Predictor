\documentclass[conference]{IEEEtran}
\IEEEoverridecommandlockouts
% The preceding line is only needed to identify funding in the first footnote. If that is unneeded, please comment it out.
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em
    T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}
\begin{document}

\title{BTC-USD Crypto-price Prediction Using LSTM\\
{\footnotesize \textsuperscript{}}
}
\author{\IEEEauthorblockN{\textsuperscript{} Michael Zats}
\IEEEauthorblockA{\textit{School of Computing, Machine Learning} \\
\textit{Prague City University/Teesside University}\\
Prague, Czechia \\
mikhail.zats@praguecollege.cz}
}

\maketitle

\begin{abstract}
Bitcoin is a cryptocurrency that has become a popular stock market investment. A variety of risk factors influence the stock market. Moreover, bitcoin is a cryptocurrency that has been steadily rising in recent years, with occasional sharp drops without any apparent impact on the stock market. Because of the volatility, there is a demand for an automated technique to forecast bitcoins on the share market. LSTM (Long Short-Term Memory) is another module supplied for RNN that was later developed and adopted by many researchers. Like RNN, the LSTM likewise comprises the module with recurrent consistency, which favours the measurement and analysis of closing prices for bitcoins as conducted in this research.
\end{abstract}

\begin{IEEEkeywords}
Machine learning, Cryptocurrencies, Bayesian neural network, Long Short-Term Memory Neural Networks, RNN, LSTM, Bitcoin prediction.
\end{IEEEkeywords}

\section{Introduction}
Unlike traditional market assets, cryptocurrency markets are incredibly volatile, and while they share many aspects of earlier stock markets, they are also extremely unstable. These marketplaces are indeed decentralized, unregulated, and prone to manipulation. Many entrepreneurs are investing in blockchain, the well-known technology that underpins the most popular cryptocurrencies, including Bitcoin, and this number is expected to expand as Bitcoin's utility grows. Many people are speculating on the bitcoin price. Making predictions on the Bitcoin price market can yield large profits, but it also has the potential to be extremely risky.
\hfill \break
\hfill \break
As a result, determining the ideal moment to reach the market is critical to maximising earnings while minimising losses.
Bitcoin's Price fluctuates daily, precisely like the Price of conventional currencies. On the other hand, Bitcoin price movements are on a much larger scale than fiat currency price swings. As a result, getting a sense of the potential price trends can be crucial. To date, various online platforms offer a variety of technical analysis tools that enable bitcoin traders to spot patterns and stock prices. Due to this impact, the number of research articles examining the future bitcoin price trends is entirely developing.
This research article proposes and investigates different machine learning-based frameworks for forecasting Bitcoin values. These frames could be used to choose when to invest and how much to invest and create bitcoin trading strategies. The primary purpose of this research is to monitor the effectiveness of Bayesian Neural Networks (BNN) in forecasting Bitcoin values to that of other types of NNs such as Feed Forward Neural Networks (FFNN) and Long Short-Term Memory Neural Networks (LSTMNN). In addition, using the strategy proposed by (Patel.et.al. 2015), we investigated if the efficiency of the FFNN and LSTMNN improves when they are used in tandem with another machine learning technique, the so-called Support Vector Regression (SVR).
\hfill \break
\hfill \break
As a result, the one-stage framework focuses on determining the daily closing price of bitcoin at a specific number of days starting from the value of these five technical analyses. Instead, the first stage, which an SVR constitutes in the two-stage frameworks, obtains the five technical analyses on particular days as input and predicts their value. According to (Patel et al.,2015), the second stage, which is constructed from one of two NNs, takes the five technical analyses as input and forecasts the closing prices price of Bitcoin (2015). The paper will focus more on LSTM as a tool for technical analysis in predicting bitcoin rates compared with USD for a certain period.
\hfill \break
\hfill \break
This research paper examines both bitcoin and stock market projections, methods, techniques, and tools using various materials, including books, papers, and other publicly available sources.

\section{Related Work}
As previously stated, the proposed frameworks, particularly the idea of a one-stage and two-stage approach, are based on Patel et al. (2015). The authors use the SVR paired with Convolutional Neural Network (ANN) together with Random Forest (RF) algorithms to forecast the future values of two Indian stock market indexes, the CNX Nifty and S&P Bombay Stock Exchange (BSE) Sensex. They compare the results achieved in these two-stage frameworks to those obtained in single-stage frameworks established by a specific ML technique, ANN, RF, and SVR, respectively. In contrast to (Patel et al.,2015), we looked into the efficiency of the BNN in the research.
The bins are considered in (Jang & Lee's 2018) work, which uses Blockchain data to forecast the log price and log volatility of Bitcoin.

\hfill \break
\hfill \break
The authors pick important inputs while analysing the multicollinearity problem, focusing on the variance inflation factor (VIF) level for each input. ( Mallqui & Fernandes ,2019) use multiple machine learning algorithms, such as recurrent neural networks, tree classifiers, and the SVM algorithm, to forecast the Price of bitcoin and its movements, thereby solving both a classification and regression problem. It was not used a selection approach because the number of inputs considered in our work was so small. It was compared the results of the different frameworks with different numbers of inputs, finding that the BNN performs best when all five technical indicators are taken into account. Our next step will be to study the effectiveness of the new frameworks with a larger number of inputs, such as blockchain information, tweet volumes, sentimental component, and the correlation related feature based on subset selection to choose the essential objectives for their prediction issue. (Mallqui & Fernandes ,2019) anticipate both the Price of bitcoin and its movements, solving both problems.
\hfill \break
\hfill \break
(Abraham.et.al,2018) combine Twitter information and Google trends information to forecast price fluctuations of Bitcoin and Ethereum. The authors use a linear model to forecast the trend of market fluctuations in this paper. (Ni.et.al,2018) looked into the predictability of cryptocurrency returns, specifically bitcoin return predictability. They used a tree-based prediction algorithm with 128 technical analyses as input to predict the bitcoin daily return. The results revealed that their predictive model had a higher speed and performance than a buy-and-hold approach. As a result, Huang, Huang, and Ni (2018) believe that technical analysis can be effective in the bitcoin market.
\hfill \break
\hfill \break
(Marchesi.et.al,2019) reached a similar conclusion by simulating the cryptocurrency pair BTC/USD trade. They create an artificial BTC/USD exchange in which Chart analysts (speculators) trade using trading strategies based on the technical indicators. In contrast, randomly based traders work without using any specific trading technique. The results reveal that chart pattern-analysers who use trading rules chosen by a genetic algorithm (GA) that optimizes its parameters can make more money.

\section{Proposed System}
Another sort of RNN module is the LSTM (Long Short-Term Memory). The LSTM network consists of recurrently consistent modules, just like the RNN. The link between the hidden units of RNN is different in LSTM, an upgraded form of RNN. The RNN's explanatory structure. The structure of RNN and LSTM is similar; the only difference is the memory cell of the structure's hidden layer. The gradient difficulties are efficiently solved by constructing three unique gates (Hager & DiPietro,2020).

\title{Fig1}
\author{Science direct}
\date{\today}

\maketitle
\begin{figure}[htp]
\centering
\includegraphics[width=5cm]{fig1.png}
\caption{RNN structure.}
\label{fig:RNN structure}
\end{figure}
RNN outperforms LSTM in performance by activating states based on the network events. A unique bias and weight make up a standard RNN node. The simple neural unit and the LSTM are used to evaluate the RNN. The network parameters are used to create a one-to-one network configuration, in which the sequence of each input data generates an output at the same time step.
The shortcomings of RNN are shown in the feedback Xo;
\( X1\) has a massive variety of data \(V(x-1), Vt, v(x+1)\), so
when the \((t+1\)) set requires information, which is already appropriate i,e \(Y(x-1), Y(x), Y(x+1)\) to RNN would unable to learn the linking of the informative data since old memory saved will become increasingly useless as time passes since it is overwritten or replaced by the new memory(Ryan,2021).

\title{Fig2}
\author{medium.com}
\date{\today}

\maketitle
\begin{figure}[htp]
\centering
\includegraphics[width=8.5cm]{fig2.png}
\caption{Single LSTM Cell.}
\label{fig:Single LSTM cell }

\end{figure}

\title{Fig3}
\author{Michael Zats}
\date{\today}

\maketitle
\begin{figure}[htp]
\centering
\includegraphics[width=8.5cm]{fig3.png}
\caption{Block Diagram of The Proposed Task.}
\label{fig:Block Diagram of The Proposed Task}

\end{figure}
According to Figure 3, The method begins with getting the data based on USD exchange rates acquired from Michaelzats /crypto-predictor and is available on GitHub (Zats,2022). This research employed real-time information on this research, with 2814 datasets in CSV format, for seven years from 2014-09-17 to 30-5-2022, which is historical data pricing. Figures 3 and 4 depict sample data from datasets.

\title{Fig4}
\author{Michael Zats}
\date{\today}

\maketitle
\begin{figure}[htp]
\centering
\includegraphics[width=8.5cm]{fig4.png}
\caption{Data sample.}
\label{fig:Data Sample}

\end{figure}

\begin{tabular}{ |p{3.65cm}|p{3.65cm}| }
\hline
\multicolumn{2}{|c|}{\textbf{Used Parameters}} \\
\hline
\textbf{Parameters}&\textbf{Description}\\
\hline
Date & Date of Asset Price\\
Open & The Opening Price of the Asst \\
Close & The Closing Price of the Asset\\
Volume & Number of Assets Traded\\
High & The Highest Asset Price of the Day\\
Low & The Lowest Asset Price of the Day\\
\hline
\end{tabular}
\hfill \break
\section{Experiential Results}

Figure 4 shows the pre-processing results for loading the dataset into the machine and algorithm, followed by the close price data for bitcoin before training, testing, and predicting the results.

\title{Fig5}
\author{Michael Zats}
\date{\today}

\maketitle
\begin{figure}[htp]
\centering
\includegraphics[width=8.5cm]{fig5.png}
\caption{Close price bitcoin data from the taken period (2020/2022-05-30).}
\label{fig:Close Price Bitcoin Data from The Taken pPeriod (2020/2022-05-30)}

\end{figure}
\hfill \break
The data was separated into two years of training and testing for both the Step Split and data training (Chinchalkar,2022). At 729 data points (2020-05-30 – 2022-05-30), the split is done (Figure 5).
It was used 70/30 pro-cent proportion to divide the data into training and testing respectively. As a result, we used the data from this year to test the dataset and the data from the previous two years to train it. Such timeframe was taken as the BTC price saw the new uptrend from that dates. Therefore, the new cycle started.
\hfill \break
\hfill \break
Before we finalise the results, we try to measure them using the mean square error (MSE). The MSE would always be greater than the MAE or equal to it. The MSE statistic assesses a model's ability to predict a linear function. The MSE measurements are the same as the dependent variable/target in the data given (if that is dollars), which helps determine whether the extent of the mistake is significant (Ependi. et al.,2019). The model's performance improves as the MSE decreases based on the production of epoch history data as a result of LSTM prediction with epoch 200. After each epoch, the training and validation loss decreases.

\title{Fig5}
\author{Michael Zats}
\date{\today}

\maketitle
\begin{figure}[htp]
\centering
\includegraphics[width=8.5cm]{fig6.png}
\caption{Training And Validation Loss Results.}
\label{fig:Training And Validation Loss Results}

\end{figure}
\hfill \break
\hfill \break
The root-mean-square deviation is also generated with the help of the python code:
\hfill \break
\hfill \break
Train data Root-mean-square deviation:\textit{1720.1624441173572},
\hfill \break
Train data Mean squared error: \textit{2958958.8341518},
\hfill \break
Train data measure of errors: \textit{1184.5165628397704},
\hfill \break
Test data Root-mean-square deviation: \textit{1687.1153282791906},
\hfill \break
Test data Mean squared error: \textit{2846358.130914601},
\hfill \break
Test data measure of errors: \textit{1282.4875955405587}
\hfill \break
\hfill \break
The values of the root mean square deviations are lower than the mean squared error, indicating the high error limitation due to the overall code performance results.

\hfill \break
Train data variance regression score: \textit{0.9901621797882803},
\hfill \break
Test data variance regression score: \textit{0.9397413351879735},
\hfill \break
Train data R2 score: \textit{0.9901618162865842},
\hfill \break
Test data R2 score: \textit{0.9338064854630307},
\hfill \break
\hfill \break
Variance regression scores and \(R-square\) scores help predict the closing price of bitcoin despite its high rate of fluctuation, which sometimes brings forth varying prediction results.

\section{Conclusion}
The proposed model successfully predicted the result of bitcoin from the GitHub dataset example. The model can produce results using time series approaches, and the findings can estimate the pricing for the coming days using the data split to train and testing what was explained on the page above. However, the downside is that the outcome is not good enough for MSE, possibly in the hundreds or later. Because the market is influenced by numerous unknown elements, such as political and economic issues at local and global levels, the Cryptocurrency exchange rates are messed up. So, LSTM prediction of bitcoin price is not good enough to decide to invest in bitcoin because it is just one side of the coin. Nonetheless, the model was also checked on stock prices, where it showed the better results with regards to MSE due to not such a big price variation. Therefore, the model can be used as one of the investments supporting tools.

\section\*{Acknowledgment}

I would like to express my special thanks of gratitude to my lecturer Petr Śvarny who gave me the golden opportunity to do this wonderful project on the topic of Machine Learning called BTC-USD Price Prediction Using LSTM , which also helped me in my own development in the field of Machine Learning and python coding I am really thankful to them.
\hfill \break
\hfill \break
Secondly i would also like to thank my parents and friends who helped me much in finalising this project within the limited time frame. That is my first research that kind, and so I am looking forward to getting better in the field of computer science writing.

\begin{thebibliography}{00}
\bibitem{b1} Abraham, J., Higdon, D., Nelson, J., & Ibarra, J. (2018). Cryptocurrency price prediction using tweet volumes and sentiment analysis. SMU Data Science Review, 1(3), 1.
\bibitem{b2} Cocco, L., Tonelli, R., & Marchesi, M. (2019). An agent-based artificial market model for studying bitcoin trading. IEEE Access, 7, 42908-42920.
\bibitem{b3} Patel, J., Shah, S., Thakkar, P., & Kotecha, K. (2015). Predicting stock market index using a fusion of machine learning techniques. Expert Systems with Applications, 42(4), 2162-2172.
\bibitem{b4} Mallqui, D. C., & Fernandes, R. A. (2019). Predicting the direction, maximum, minimum and closing prices of daily Bitcoin exchange rate using machine learning techniques. Applied Soft Computing, 75, 596-606.
\bibitem{b5} Huisu, J., Lee, J., Ko, H., & Lee, W. (2018, August). Predicting bitcoin prices by using the rolling window LSTM model. In Proceedings of the KDD data science in Fintech Workshop, London, UK (pp. 19-23).
\bibitem{b6} Chinchalkar. (2022, May 11). Bloomberg. Bloomberg - Are you a robot?. https://www.bloomberg.com/news/articles/2022-05-11/history-of-bitcoin-slumps-makes-20-000-realistic-target-chart.
\bibitem{b7} Hager & DiPietro. (2020). Recurrent neural network. ScienceDirect.com | Science, health and medical journals, full-text articles and books. https://www.sciencedirect.com/topics/engineering/recurrent-neural-network
\bibitem{b8} Ryan T. J. J. (2021, September 10). LSTMs explained: A complete, technically accurate, conceptual guide with Keras. Medium. https://medium.com/analytics-vidhya/lstms-explained-a-complete-technically-accurate-conceptual-guide-with-keras-2a650327e8f2
\bibitem{b9} Ependi.et.al. (2019). An LSTM-Method for Bitcoin Price Prediction: A Case Study Yahoo Finance Stock Market. International Conference on Electrical engineering and Computer Science, 206-210.
\bibitem{b10} Zats.M (2022). Crypto-predictor/Github
\bibitem{b11} Huang, J. Z., Huang, W., & Ni, J. (2019). Predicting bitcoin returns using high-dimensional technical indicators. The Journal of Finance and Data Science, 5(3), 140-155.
\end{thebibliography}

\end{document}
