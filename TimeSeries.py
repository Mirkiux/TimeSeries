import numpy as np
import statsmodels.api as statsmodels
from statsmodels.tsa.arima.model import ARIMA

def create_hankel_matrix(time_series,m,L):
    time_series = np.array(time_series)
    time_series = time_series[-L:]
    print(time_series[5])
    hankel_matrix_cols = L-m+1
    hankel_matrix_rows = m
    #indexes_matrix = np.zeros((hankel_matrix_rows, hankel_matrix_cols)) 
    hankel_matrix = np.zeros((hankel_matrix_rows, hankel_matrix_cols))
    for col in range(hankel_matrix_cols):
        initial_index = col
        for row in range(hankel_matrix_rows):
            hankel_matrix[row,col] = time_series[initial_index + row]
            #indexes_matrix[row,column] = (p+row-(column + 1))
    return hankel_matrix

def build_page_matrix(time_series,L):
    time_series = np.array(time_series)
    T = len(time_series)
    #compute len to use as the T minus the remainder of T/L
    T_used = T - T%L
    time_series_to_use = time_series[-T_used:]
    page_matrix = np.reshape(time_series_to_use, (L, -1), order = 'F')
    return page_matrix


def page_matrix(X, L):
    T = len(X) 
    return X.reshape(L, T//L, order='F')

def reconstruct_matrix_threshold(time_series,m,L,energy_threshold = 0.99):
    hankel_matrix = create_hankel_matrix(time_series,m,L)
    num_values = np.count_nonzero(hankel_matrix != 0)
    P_bar = num_values/(hankel_matrix.shape[0]*hankel_matrix.shape[1])  
    U, s, Vt = np.linalg.svd(hankel_matrix)
    #printing singular values
    print(f"valores singulares: {s}")
    V = Vt.T
    # getting singular values with accumulated energy of 90%
    s2 = s**2
    energy = np.cumsum(s2)/np.sum(s2)
    print(f"energia acumulada: {energy}")
    K90 = np.argmax(energy > energy_threshold) + 1 
    print(f"K90: {K90}")
    #truncate components
    s_truncated = s[:K90]
    X_reconstructed = (1/P_bar)* np.dot(U[:, :K90], np.dot(np.diag(s_truncated), Vt[:K90, :]))
    return X_reconstructed

def reconstruct_matrix_num_svalues(time_series,m,L,num_svalues = 2):
    hankel_matrix = create_hankel_matrix(time_series,m,L)
    num_values = np.count_nonzero(hankel_matrix != 0)
    P_bar = num_values/(hankel_matrix.shape[0]*hankel_matrix.shape[1]) 
    U, s, Vt = np.linalg.svd(hankel_matrix)
    #printing singular values
    print(f"valores singulares: {s}")
    V = Vt.T
    # getting singular values with accumulated energy of 90%
    s2 = s**2
    energy = np.cumsum(s2)/np.sum(s2)
    print(f"energia acumulada: {energy}")
    K90 = num_svalues
    print(f"K90: {K90}")
    #truncate components
    s_truncated = s[:num_svalues]
    X_reconstructed = (1/P_bar)* np.dot(U[:, :K90], np.dot(np.diag(s_truncated), Vt[:K90, :]))
    return X_reconstructed

def get_non_stationary_component(time_series,m,L):
    time_series = np.array(time_series)
    hankel_matrix = create_hankel_matrix(time_series, m, L)
    reconstructed_hankel_matrix = reconstruct_matrix_num_svalues(time_series, m, L, num_svalues = 2)
    #get last row of hankel matrix
    last_row = hankel_matrix[-1,:].reshape(-1,1)
    print(last_row.shape)
    #remove last row from reconstructed hankel matrix
    reconstructed_hankel_matrix = reconstructed_hankel_matrix[:-1,:].transpose()
    print(reconstructed_hankel_matrix.shape)

    #execute linear regression without constant using statsmodels.regression.linear_model.OLS
    model = statsmodels.regression.linear_model.OLS(last_row, reconstructed_hankel_matrix)
    model = model.fit()
    print(model.summary())
    #get residuals
    residuals = model.resid
    #getting fitted values
    #fitted_values = model.fittedvalues
    return model, residuals, hankel_matrix, reconstructed_hankel_matrix

def one_period_prediction(time_series,m,L):
    time_series = np.array(time_series)
    model, residuals, _, _ = get_non_stationary_component(time_series,m,L)
    #get last column of reconstructed hankel matrix
    last_values = time_series[-(m-1):]
    print(last_values)
    non_stationary_prediction = float(model.predict(last_values))
    #arima model with d=0, q= 0 and p=3 and trend = 'n'
    arima_model = ARIMA(residuals, order=(3,0,0), trend='n')
    arima_model = arima_model.fit()
    arima_prediction = float(arima_model.forecast(steps=1))
    prediction = non_stationary_prediction + arima_prediction
    return non_stationary_prediction,arima_prediction,prediction

def multi_period_prediction(time_series,m,L,periods):
    time_series = np.array(time_series)
    non_stationary_predictions = []
    arima_predictions = []
    predictions = []
    for i in range(periods):
        non_stationary_prediction,arima_prediction,prediction = one_period_prediction(time_series,m,L)
        predictions.append(prediction)
        non_stationary_predictions.append(non_stationary_prediction)
        arima_predictions.append(arima_prediction)
        time_series = np.append(time_series,prediction)
    return {'non_stationary_predictions':non_stationary_predictions,'arima_predictions':arima_predictions,'predictions':predictions}

def compute_empirical_autocovariance(time_series,tao):
    if tao < 0 :
        return compute_empirical_autocovariance(time_series,-tao)
    else:
        # Reshape the data to fit the scaler
        time_series = np.array(time_series).reshape(-1, 1)
        # Subtract the mean from the time series
        centered_time_series = time_series - np.mean(time_series)
        
        current_time_series = centered_time_series[:len(centered_time_series) - tao]
        shifted_time_series = centered_time_series[tao:]
        R_tao = np.sum(current_time_series * shifted_time_series)/len(time_series)

        return R_tao

def compute_empirical_autocorrelation(time_series,tao):
    return compute_empirical_autocovariance(time_series,tao)/compute_empirical_autocovariance(time_series,0)
