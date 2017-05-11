import pandas as pd
import matplotlib.pyplot as plt
#learning rate
lr = .0001
count = -1

def err(b,m,data):
    total_err = 0
    for i in range(0,len(data)):
        x = data.iat[i,0]
        y = data.iat[i,1]
        total_err += (y - (m * x + b))**2
    return total_err/float(len(data))

def ev_and_fw(b_curr, m_curr, data):
    #evaluate partial
    m_gradient = 0
    b_gradient = 0
    N = float(len(data))
    for i in range(0,len(data)):
        x = data.iat[i,0]
        y = data.iat[i,1]
        #greatest ascent:
        m_gradient += (-2/N) * x * (y - ((m_curr * x) + b_curr))
        b_gradient += (-2/N) * (y - ((m_curr * x) + b_curr))
    #Subtract from current values to move towards minimum
    new_m = m_curr - (m_gradient *lr)
    new_b = b_curr - (b_gradient *lr)
    return [new_b, new_m]

def connector(b_start, m_start, iters, count, data):
    print('Reducing error...')
    b = b_start
    m = m_start
    for i in range(iters):
        count += 1
        b, m = ev_and_fw(b,m, data)
        if (count % 50 == 0) or (count == 0):
            print(err(b,m, data))
    return [b, m]

def run():
    #get data
    data = pd.read_csv('data.csv', header=None)
    #number of iterations
    iters = 1000
    m_i = 0
    b_i = 0
    ini_err = err(m_i,b_i,data)

    final_b, final_m = connector(b_i, m_i, iters, count, data)

    final_err = err(final_b, final_m, data)
    pct_err_red = ini_err - (final_err / ini_err)

    print('Final optimal values b= {0}, m = {1} and the final standard error {2}. The error has been reduced by {3}%.'.format(final_b,final_m,final_err,int(pct_err_red)))

    x_ = data.iloc[:,0]
    y_ = data.iloc[:,1]
    plt.scatter(x_,y_)

    xx = [int(i) for i in range(0,99)]
    yy = [i * final_m + final_b for i in xx]
    plt.plot(xx,yy)
    plt.show

if __name__ == '__main__':
    run()
