import numpy as np

class Quaternion:

    """
    returns Q matrix of a quaternion. An anti-symmetric orthogonal 
    matrix that helps with quaternion multiplication. 
    See http://www.diva-portal.org/smash/get/diva2:1010947/FULLTEXT01.pdf
    """
    @staticmethod
    def Q_Matrix(q):
        q0 = q[0]
        q1 = q[1]
        q2 = q[2]
        q3 = q[3]
        return np.array([  [q0, -q1, -q2, -q3],
                           [q1, q0, -q3, q2],
                           [q2, q3, q0, -q1],
                           [q3, -q2, q1, q0]])
    """
    Returns Quaternion : the conjugate of the quaternion in question
    """
    @staticmethod
    def conjugate(q):
        q0 = q[0]
        q1 = q[1]
        q2 = q[2]
        q3 = q[3]
        return np.array([q0,-q1,-q2,-q3])

    """
    Returns rotation matrix of quaternion.
    """
    @staticmethod
    def rotationMatrix(q):
        q0 = q[0]
        q1 = q[1]
        q2 = q[2]
        q3 = q[3]
        R = np.zeros([3,3])

        R[0,0] = q0**2 + q1**2 - q2**2 - q3**2
        R[1,0] = 2*(q1*q2 + q0*q3)
        R[2,0] = 2*(q1*q3 - q0*q2)

        R[0,1] = 2*(q1*q2 - q0*q3)
        R[1,1] = q0**2 - q1**2 + q2**2 - q3**2
        R[2,1] = 2*(q2*q3 + q0*q1)

        R[0,2] = 2*(q1*q3 + q0*q2)
        R[1,2] = 2*(q2*q3 - q0*q1)
        R[2,2] = q0**2 - q1**2 - q2**2 + q3**2
        return R
    """
    q : List_like object that contains 4 elements e.g. ndarray, list, or tupple
    """
    def __init__(self, q):
        if len(q) != 4:
            raise Exception("Quaternion must have 4 real elements")
        self.q = np.array([q[0],q[1],q[2],q[3]])
        self.C = Quaternion.conjugate(self.q)
        self.Q = Quaternion.Q_Matrix(self.q)
        self.R = Quaternion.rotationMatrix(self.q)

    """
    Returns Length of the quaternion. Must always be 4.
    """
    def __len__(self):
        return len(self.q)
    def __str__(self) -> str:
        return "Quaternion(["+str(self[0])+","+str(self[1])+","+str(self[2]) +","+ str(self[3])+"])"
    def __repr__(self):
        return "Quaternion(["+str(self[0])+","+str(self[1])+","+str(self[2]) +","+ str(self[3])+"])"
    def __getitem__(self,key):
        return self.q[key]
    

    def __mul__(self,other):
        if not isinstance(other, (float, int)):
            raise TypeError("Right operand should be a numerical scalar value. It can't be " + str(type(other)))
        return Quaternion(other * self.q)
    def __rmul__(self,other):
        if not isinstance(other, (float, int)):
            raise TypeError("Left operand should be a numerical scalar value. It can't be " + str(type(other)))
        return Quaternion(other * self.q)

        
    """
    Quaternion multiplication. Plain and simple.
    """
    def __matmul__(self,other):
        if len(other) != 4:
            raise Exception("Right operand quaternion must have 4 elements")
        result = np.dot(self.Q, other.q)
        return Quaternion(result)
    """
    Returns the norm of the quaternion.
    """
    def norm(self):
        return np.linalg.norm(self.q)
    

    """
    Returns unit quaternion of the current quaternion.
    """
    def normalize(self):
        return Quaternion(self.q/self.norm())


    
    """
    Returns the inverse of the quaternion. q*q_inv = [1,0,0,0] (unit quaternion)
    """
    def inv(self):
        return Quaternion(self.C / (self.norm() ** 2))

    """
    Rotates vector, or any point in affine space, and expressing its image in a fixed reference frame

    Input v: an array-like containing 3 elements representing the vector expressed in a fixed frame.
    Returns: rotation of v in the fixed frame by the used quaternion.
    """
    def rotate(self,v):
        if len(v) != 3:
            raise ValueError("Vector v must have (3) elements. But it in reality, it had : ("+ str(len(v))+  ") elements")
        
        return self.R @ v
    """
    Transforms vector v to another reference frame, represented by quaternion q.
    """
    def transform(self,v):
        if len(v) != 3:
            raise ValueError("Vector v must have (3) elements. But it in reality, it had : ("+ str(len(v))+  ") elements")
        
        return self.R.T @ v        
