from __future__ import print_function

''' 
Preprocess audio
'''
import numpy as np
import librosa
import librosa.display
import os

def get_class_names(path="Audio/"):  # class names are subdirectory names in Samples/ directory
    class_names = os.listdir(path)
    return class_names

def preprocess_dataset(inpath="Audio/", outpath="Preproc/"):

    if not os.path.exists(outpath):
        os.mkdir( outpath, 0o0755 );   # make a new directory for preproc'd files

    class_names = get_class_names(path=inpath)   # get the names of the subdirectories
    nb_classes = len(class_names)
    print("class_names = ",class_names)
    for idx, classname in enumerate(class_names):   # go through the subdirs

        if not os.path.exists(outpath+classname):
            os.mkdir( outpath+classname, 0o0755 );   # make a new subdirectory for preproc class

        class_files = os.listdir(inpath+classname)
        n_files = len(class_files)
        n_load = n_files
        print(' class name = {:14s} - {:3d}'.format(classname,idx),
            ", ",n_files," files in this class",sep="")

        printevery = 20
        for idx2, infilename in enumerate(class_files):
            audio_path = inpath + classname + '/' + infilename
            if (0 == idx2 % printevery):
                print('\r Loading class: {:14s} ({:2d} of {:2d} classes)'.format(classname,idx+1,nb_classes),
                       ", file ",idx2+1," of ",n_load,": ",audio_path,sep="")
            #start = timer()
            aud, sr = librosa.load(audio_path, sr=None)
            melgram = librosa.amplitude_to_db(librosa.feature.melspectrogram(aud, sr=sr, n_mels=96),
                                          ref=1.0)[np.newaxis,np.newaxis,:,:]
            #melgram = librosa.amplitude_to_db(librosa.feature.mfcc(y=aud, sr=sr, n_mfcc=40),
            #                               ref=1.0)[np.newaxis,np.newaxis,:,:]
            # potential different representation
            # stft = np.abs(librosa.stft(X))
            # mfccs = librosa.feature.mfcc(y=aud, sr=sr, n_mfcc=40)
            # chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            # mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            # contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
            # tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
            outfile = outpath + classname + '/' + infilename+'.npy'
            #print(melgram)
            if(melgram.shape[3] < 100):
                melgram_segmentation = np.copy(melgram)
                melgram_segmentation = np.resize(melgram_segmentation,(1,1,96,100))
                np.save(outfile,melgram_segmentation)
            else:
                if(melgram.shape[3] % 100 == 0):
                    segments_num = melgram.shape[3]//100
                else:
                    segments_num = melgram.shape[3]//100 + 1
                melgram_segmentation = np.copy(melgram)
                melgram_segmentation = np.resize(melgram_segmentation,(1,1,96,100*int(segments_num)))
                #print (segments_num)
                for i in range(segments_num):
                    outfile_segmented = outpath + classname + '/' +str(i)+'-'+ infilename+'.npy'
                    melgram_segment = melgram_segmentation[:,:,:,100*i:100*(i+1)]
                    np.save(outfile_segmented,melgram_segment)


if __name__ == '__main__':
    preprocess_dataset()
