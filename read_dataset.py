import numpy as np
import scipy.io as sio
import wget
import h5py
import os
import argparse


rootOutputDir = './dataset'
#Dataset information (Arad and Ben-Shahar, Sparse Recovery of Hyperspectral Signal from Natural RGB Images, ECCV, 2016)
datasetRoot = "/media/omnisky/7D37935326D33C41/wmy/data/optimized-spectral-SR_dataset/"
listOfParkFiles = ['omer_0331-1055.mat', 'omer_0331-1102.mat', 'omer_0331-1104.mat', 'omer_0331-1135.mat', 'omer_0331-1159.mat', 'prk_0328-1025.mat', 'prk_0328-1031.mat', 'prk_0328-1034.mat', 'prk_0328-1045.mat']
listOfRuralFiles = [ 'eve_0331-1647.mat', 'eve_0331-1656.mat', 'eve_0331-1657.mat', 'eve_0331-1705.mat', 'rsh_0406-1413.mat']

fractionTrain = 0.7
numSpectraPerFile = 2000

def main():
    parser = argparse.ArgumentParser(description='Download Arad and Ben-Shahar, ECCV, 2016 dataset:')
    parser.add_argument('--train', type=str, default='park', help='Training subset (default: park)')
    parser.add_argument('--test', type=str, default='rural', help='Testing subset (default: rural)')
    args = parser.parse_args()

    subsetMap = {'park': listOfParkFiles, 'rural': listOfRuralFiles}
    trainSubset = subsetMap[args.train]
    testSubset = subsetMap[args.test]

    if not os.path.exists(rootOutputDir):    os.mkdir(rootOutputDir)
    if not os.path.exists(os.path.join(rootOutputDir,'train')):    os.mkdir(os.path.join(rootOutputDir,'train'))
    if not os.path.exists(os.path.join(rootOutputDir,'val')):    os.mkdir(os.path.join(rootOutputDir,'val'))
    if not os.path.exists(os.path.join(rootOutputDir,'test')):    os.mkdir(os.path.join(rootOutputDir,'test'))
    

    #Download training and validation spectra
    spectra = []
    for i in range(len(trainSubset)):
        fileurl = datasetRoot + trainSubset[i]
        print(trainSubset[i])
        data = h5py.File(fileurl,  'r')
        HSBands = data["bands"][()].ravel().tolist()

        if i==0:
            np.savetxt(os.path.join(rootOutputDir,'HSBands.csv'), HSBands) 

        spectrum = data["rad"][()].astype(np.float32)
        spectrum = np.transpose(spectrum, [1,2,0])
        spectrum = np.reshape(spectrum, [-1, len(HSBands)]) 
        spectra.append(spectrum)

    spectra = np.vstack(spectra)
    spectra = spectra[np.random.permutation(spectra.shape[0]),:]
   

    numTrain = spectra.shape[0]
    trainSpectra = spectra[:int(numTrain*fractionTrain)]
    valSpectra = spectra[int(numTrain*fractionTrain):]

    for i in range(0,trainSpectra.shape[0]-numSpectraPerFile,numSpectraPerFile):
        sp = trainSpectra[i:(i+numSpectraPerFile),:]
        np.save(os.path.join(rootOutputDir, 'train', 'train_chip%d.npy'%i), sp)

    for i in range(0,valSpectra.shape[0]-numSpectraPerFile,numSpectraPerFile):
        sp = valSpectra[i:(i+numSpectraPerFile),:]
        np.save(os.path.join(rootOutputDir, 'val', 'val_chip%d.npy'%i), sp)



if __name__=='__main__':
    main()
