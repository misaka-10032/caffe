INPUT = '/Users/rocky/Research/caffe/examples/xiyuan/test/output.h5'
df = pd.read_hdf(INPUT, 'df')
scores = np.array([df['prediction'].values[i][0] for i in xrange(len(df['prediction']))])
print scores
