# An Analysis of SVD for Deep Rotation Estimation

This repository will contain code for experiments in an upcoming NeurIPS 2020 paper:

**An Analysis of SVD for Deep Rotation Estimation** \
Jake Levinson, Carlos Esteves, Kefan Chen, Noah Snavely, Angjoo Kanazawa, Afshin Rostamizadeh, and Ameesh Makadia \
To appear in the *34th Conference on Neural Information Processing Systems
(NeurIPS 2020)*. \
[arXiv](https://arxiv.org/abs/2006.14616)


See the [**special_orthogonalization** repository under **google_research**](https://github.com/google-research/google-research/tree/master/special_orthogonalization) which hosts the experiments we wrote in TensorFlow.

### Sample Code
Below is sample code to use SVD orthogonalization to generate 3D rotation
matrices from 9D inputs.


**TensorFlow:**

```python
def symmetric_orthogonalization(x):
  """Maps 9D input vectors onto SO(3) via symmetric orthogonalization.

  x: should have shape [batch_size, 9]

  Returns a [batch_size, 3, 3] tensor, where each inner 3x3 matrix is in SO(3).
  """
  m = tf.reshape(x, (-1, 3, 3))
  _, u, v = tf.svd(m)
  det = tf.linalg.det(tf.matmul(u, v, transpose_b=True))
  r = tf.matmul(
      tf.concat([u[:, :, :-1], u[:, :, -1:] * tf.reshape(det, [-1, 1, 1])], 2),
      v, transpose_b=True)
  return r
```



**PyTorch:**

```python
def symmetric_orthogonalization(x):
  """Maps 9D input vectors onto SO(3) via symmetric orthogonalization.

  x: should have size [batch_size, 9]

  Output has size [batch_size, 3, 3], where each inner 3x3 matrix is in SO(3).
  """
  m = x.view(-1, 3, 3)
  u, s, v = torch.svd(m)
  vt = torch.transpose(v, 1, 2)
  det = torch.det(torch.matmul(u, vt))
  det = det.view(-1, 1, 1)
  vt = torch.cat((vt[:, :2, :], vt[:, -1:, :] * det), 1)
  r = torch.matmul(u, vt)
  return r
```

# Experiments

## TensorFlow experiments
For the experiments which we (re)implemented in TensorFlow, the [code is available in a google_research package](https://github.com/google-research/google-research/tree/master/special_orthogonalization).


## Inverse Kinematics
This is the experiment reported in Section 4.4.2 and Table 6 of the paper.

The experiment is presented originally in [Yi Zhou et al, CVPR19](https://arxiv.org/abs/1812.07035), and the authors' code can be found [here](https://github.com/papagina/RotationContinuity/tree/master/Inverse_Kinematics). \
**The few changes required to run the training and eval with SVD orthogonalization are available here:** \
[Edits to IK experiment for SVD orthogonalization](https://github.com/amakadia/RotationContinuity)

## Single Image Depth prediction on KITTI
This is the experiment reported in Section 4.4.3 of and Table 7 of the paper.

The experiment is presented originally in [Tinghui Zhou et al, CVPR17](https://people.eecs.berkeley.edu/~tinghuiz/projects/SfMLearner/), and the author's code can be found [here](https://github.com/tinghuiz/SfMLearner). \
**We modified the pose layer to replace the Euler angle representation with SVD orthogonalization. See the changes needed to run the model with SVD orthogonalization:** \
[Replacing Euler angles with SVD orthogonalization in SfmLearner](https://github.com/amakadia/SfMLearner)

# Citation

```
@inproceedings{levinson20neurips,
  title = {An Analysis of {SVD} for Deep Rotation Estimation},
  author = {Jake Levinson, Carlos Esteves, Kefan Chen, Noah Snavely, Angjoo Kanazawa, Afshin Rostamizadeh, and Ameesh Makadia},
  booktitle = {Advances in Neural Information Processing Systems 34},
  year = {2020},
  note = {To appear in}
}
```
