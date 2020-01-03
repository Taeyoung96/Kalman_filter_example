import numpy as np
import matplotlib.pyplot as plt
import math

class KalmanFilter(object):
    def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):

        if(F is None or H is None):
            raise ValueError("Set proper system dynamics.")

        self.n = F.shape[1]
        self.m = H.shape[1]

        self.F = F
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

    def predict(self, u = 0):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x  

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), 
        	(I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)

def example():
	dt = 1.0/60
	F = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
	H = np.array([1, 0, 0]).reshape(1, 3)
	Q = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])
	R = np.array([0.5]).reshape(1, 1)	

	x = np.linspace(0, 100, 100)
	measurements_x = x + np.random.normal(0, 2, 100)
	y = 2*np.linspace(0, 100, 100)
	measurements_y = y + np.random.normal(0, 2, 100)

	kf_x = KalmanFilter(F = F, H = H, Q = Q, R = R)
	kf_y = KalmanFilter(F = F, H = H, Q = Q, R = R)
	kf_d = KalmanFilter(F = F, H = H, Q = Q, R = R)
	
	predictions_x = []
	predictions_y = []
	distance = []
	measurements_d = []
	predictions_d = []

	for z in measurements_x:
		predictions_x.append(np.dot(H,  kf_x.predict())[0])
		kf_x.update(z)

	plt.plot(range(len(x)),x,label = 'GroundTruth X')
	plt.plot(range(len(measurements_x)), measurements_x, label = 'Measurements X')
	plt.plot(range(len(predictions_x)), np.array(predictions_x), label = 'Kalman Filter Result X')

	for z in measurements_y:
		predictions_y.append(np.dot(H,  kf_y.predict())[0])
		kf_y.update(z)

	plt.plot(range(len(y)),y,label = 'GroundTruth Y')
	plt.plot(range(len(measurements_y)), measurements_y, label = 'Measurements Y')
	plt.plot(range(len(predictions_y)), np.array(predictions_y), label = 'Kalman Filter Result Y')


	for i in range(0,100):
		distance.insert(i,math.sqrt(math.pow(x[i],2)+math.pow(y[i],2)))
		measurements_d.insert(i,math.sqrt(math.pow(measurements_x[i],2)+math.pow(measurements_y[i],2)))

	for z in measurements_d:
		predictions_d.append(np.dot(H,  kf_d.predict())[0])
		kf_d.update(z)

	plt.plot(range(len(distance)),distance,label = 'GroundTruth D')
	plt.plot(range(len(measurements_d)),np.array(measurements_d),label = 'Measurement D')
	plt.plot(range(len(predictions_d)),np.array(predictions_d),label = 'Kalman Filter Result D')
	plt.xlabel("time(s)")
	plt.ylabel("distance(m)")
	plt.legend()
	plt.show()

if __name__ == '__main__':
    example()