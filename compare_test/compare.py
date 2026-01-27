import os,glob,sys,torchaudio,torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import scipy.io.wavfile as Wavfile 
import numpy as np
from model import XVADModel

#加载 ten 、silero,如果没有就clone 
silero_path = '/hpc_stor03/sjtu_home/zhiqiang.yin/project/model_eva/silero-vad/src'#add your silero,
ten_path ='/hpc_stor03/sjtu_home/zhiqiang.yin/project/model_eva/ten-vad/include'# add your ten 
if not os.path.exists(silero_path):
    os.system('git clone https://github.com/snakers4/silero-vad.git')#clone silero vad repo
    silero_path = "./silero-vad/src"
if not os.path.exists(ten_path):
    os.system('git clone https://github.com/TEN-framework/ten-vad.git')# clone ten vad repo
    ten_path = './ten-vad/include'
sys.path.append(silero_path)
sys.path.append(ten_path)

from silero_vad.utils_vad import VADIterator,init_jit_model
from ten_vad import TenVad

def convert_xlabel(label_file,hop_size=160,win_size=400):
    with open(label_file, "r") as f:
        lines = f.readlines()
    content = lines[0].strip().split(",")[1:]
    start = np.array(
        content[::3], dtype=float
    )  # Start point of each audio segment
    end = np.array(
        content[1:][::3], dtype=float
    )  # End point of each audio segment
    lab_manual = np.array(
        content[2:][::3], dtype=int
    )  # label, 0/1 stands for non-speech or speech, respectively
    assert (
        len(start) == len(end) 
        and len(start) == len(lab_manual) 
        and len(end) == len(lab_manual)
    )
    sample_len = int((end[-1]-start[0])*16000)
    label_frame = np.zeros(sample_len)
    label_final = np.zeros(int(sample_len/hop_size//2))
    for start_id,end_id,label_ma in zip(start,end,lab_manual):
        if label_ma == 1:
            label_frame[int(start_id*16000):int(end_id*16000)]=1
    for id,i in enumerate(range(0,len(label_frame),hop_size*2)):
        if id>=len(label_final):break
        ratio1 = float(np.mean(label_frame[i:i+win_size]))
        ratio2 =float(np.mean(label_frame[i+hop_size:i+hop_size+win_size]))
        ratio = (ratio1+ratio2)/2
        if ratio > 0.5:
            label_final[id] = 1
    return label_final


def convert_label_to_framewise(label_file, hop_size):
    frame_duration = hop_size / 16000
    with open(label_file, "r") as f:
        lines = f.readlines()
    content = lines[0].strip().split(",")[1:]
    start = np.array(
        content[::3], dtype=float
    )  # Start point of each audio segment
    end = np.array(
        content[1:][::3], dtype=float
    )  # End point of each audio segment
    lab_manual = np.array(
        content[2:][::3], dtype=int
    )  # label, 0/1 stands for non-speech or speech, respectively
    assert (
        len(start) == len(end) 
        and len(start) == len(lab_manual) 
        and len(end) == len(lab_manual)
    )
    
    num = np.array(
        np.round(((end - start) / frame_duration)), dtype=np.int32
    )  # get number of frames of each audio segment
    label_framewise = np.array([])
    for segment_idx in range(len(num)):
        cur_lab = int(lab_manual[segment_idx])
        num_segment = num[segment_idx]

        if cur_lab == 1:
            vad_result_this_segment = np.ones(num_segment)
        elif cur_lab == 0:
            vad_result_this_segment = np.zeros(num_segment)
        label_framewise = np.append(label_framewise, vad_result_this_segment)
    frame_num = min(
        label_framewise.__len__(), int((end[-1] - start[0]) / frame_duration)
    )
    label_framewise = label_framewise[:frame_num]

    return label_framewise

def get_precision_recall(VAD_result,label,threshold):
    vad_result_new = np.where(VAD_result>=threshold,1,0)
    TN, FP, FN, TP = confusion_matrix(label, vad_result_new).ravel()
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    FNR = FN / (TP + FN) if (TP + FN) > 0 else 0  
    return precision,recall,FPR,FNR  

def xvad_infer_single_file(wav_path,model):
    seg_duration = 3*16000
    wav,sr = torchaudio.load(wav_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    total_len = wav.shape[1]
    vad_result = np.array([])
    trans = torchaudio.transforms.MelSpectrogram(sample_rate=16000,n_fft=400,hop_length=160,win_length=400,n_mels=80)
    for i in range(0,total_len,seg_duration):
        if wav[:,i:i+seg_duration].shape[1]<seg_duration: break
        spec = trans(wav[:,i:i+seg_duration])
        result,_ = model(spec)
        result = result.squeeze(0).T.detach().numpy()
        vad_result = np.append(vad_result,result)
    return vad_result

def silero_vad_inference_single_file(wav_path):
    model = init_jit_model(os.path.join(silero_path,'silero_vad/data/silero_vad.jit'))
    vad_iterator = VADIterator(model)
    window_size_samples = 512
    speech_probs = np.array([])
    
    wav, sr = torchaudio.load(wav_path)
    wav = wav.squeeze(0)
    for i in range(0, len(wav), window_size_samples):
        chunk = wav[i: i+ window_size_samples]
        if len(chunk) < window_size_samples:
            break
        speech_prob = model(chunk, sr).item()
        speech_probs = np.append(speech_probs, speech_prob)
    vad_iterator.reset_states()  # reset model states after each audio
    
    return speech_probs, window_size_samples

def ten_vad_process_wav(ten_vad_instance, wav_path, hop_size=256):
    _, data = Wavfile.read(wav_path)
    num_frames = data.shape[0] // hop_size
    voice_prob_arr = np.array([])
    for i in range(num_frames):
        input_data = data[i * hop_size: (i + 1) * hop_size]
        voice_prob, _ = ten_vad_instance.process(input_data)
        voice_prob_arr = np.append(voice_prob_arr, voice_prob)
    return voice_prob_arr

if __name__ == "__main__":
    test_dir = "testset" # test wavs dir
    label_all_wav ,vad_result_ten_all = np.array([]),np.array([])
    label_hop_512_all,vad_result_silero_all = np.array([]),np.array([])
    label_x_all,vad_result_x_all = np.array([]),np.array([])

    hop_size = 256
    threshold = 0.5
    wav_list = glob.glob(f"{test_dir}/*.wav")

    print("start processing")
    for wav_path in wav_list:
        #run ten vad
        ten_vad_instance = TenVad(hop_size,threshold)
        labelfile = wav_path.replace(".wav",".scv")
        label = convert_label_to_framewise(labelfile,hop_size)
        vad_result_ten = ten_vad_process_wav(ten_vad_instance,wav_path,hop_size)
        frame_num = min(len(label),len(vad_result_ten))
        vad_result_ten_all = np.append(vad_result_ten_all,vad_result_ten[1:frame_num])
        label_all_wav = np.append(label_all_wav,label[:frame_num-1]) #different
        del ten_vad_instance

        #run silero
        label_hop_512 = convert_label_to_framewise(labelfile,hop_size=512)
        vad_result_silero,_ =silero_vad_inference_single_file(wav_path)
        frame_num = min(len(label_hop_512),len(vad_result_silero))
        vad_result_silero_all = np.append(vad_result_silero_all,vad_result_silero[:frame_num])
        label_hop_512_all = np.append(label_hop_512_all,label_hop_512[:frame_num])

        #run xvad
        xmodel = XVADModel()
        checkpoint = "../checkpoints/xvad_epoch_5.pth"
        state_dict = torch.load(checkpoint)
        xmodel.load_state_dict(state_dict)
        xmodel.eval()
        xvad_result = xvad_infer_single_file(wav_path,xmodel)
        label_xvad = convert_xlabel(labelfile,hop_size=160,win_size=400)
        frame_num = min(len(xvad_result),len(label_xvad))
        label_x_all = np.append(label_x_all,label_xvad[:frame_num])
        vad_result_x_all = np.append(vad_result_x_all,xvad_result[:frame_num])

    # Compute Precision and Recall  
    threshold_arr = np.arange(0, 1.01, 0.01)
    pr_data_ten_arr = np.zeros((threshold_arr.__len__(), 3))
    pr_data_silero_arr = np.zeros((threshold_arr.__len__(), 3))
    pr_data_xvad_arr = np.zeros((threshold_arr.__len__(), 3))
    for ind, threshold in enumerate(threshold_arr):
        precision, recall, FPR, FNR = get_precision_recall(vad_result_ten_all, label_all_wav, threshold)
        pr_data_ten_arr[ind] = precision, recall, threshold

        precision_silero_vad, recall_silero_vad, FPR_silero_vad, FNR_silero_vad = get_precision_recall(vad_result_silero_all, label_hop_512_all, threshold)
        pr_data_silero_arr[ind] = precision_silero_vad, recall_silero_vad, threshold
        
        xprecision, recall, FPR, FNR = get_precision_recall(vad_result_x_all, label_x_all, threshold)
        pr_data_xvad_arr[ind] = xprecision, recall, threshold

        if threshold in np.arange(0,1,0.1):
            print(f"when threshold={threshold}\nten vad:{precision},silero vad:{precision_silero_vad},xvad:{xprecision}")
    
    # Plot PR Curve
    print("Plotting PR Curve")
    pr_data_arr_to_plot = pr_data_ten_arr[:-1] 
    plt.plot(
        pr_data_arr_to_plot[:, 1],
        pr_data_arr_to_plot[:, 0],
        color="red",
        label="TEN VAD",
    )  # Precision on y-axis, Recall on x-axis
    pr_data_silero_vad_arr_to_plot = pr_data_silero_arr[:-1]
    plt.plot(
        pr_data_silero_vad_arr_to_plot[:, 1],  # Recall (x-axis)
        pr_data_silero_vad_arr_to_plot[:, 0],  # Precision (y-axis)
        color="blue",
        label="Silero VAD",
    )
    pr_data_xvad_arr_to_plot = pr_data_xvad_arr[:-1]
    plt.plot(
        pr_data_xvad_arr_to_plot[:, 1],  # Recall (x-axis)
        pr_data_xvad_arr_to_plot[:, 0],  # Precision (y-axis)
        color="yellow",
        label=" XVAD",
    )

    plt.xlabel("Recall", fontsize=14, fontweight="bold", color="black")
    plt.ylabel("Precision", fontsize=14, fontweight="bold", color="black") 
    legend = plt.legend()
    legend.get_texts()[0].set_fontweight("bold")
    legend.get_texts()[1].set_fontweight("bold")
    plt.grid(True)
    plt.xlim(0.65, 1)
    plt.ylim(0.6, 1)
    plt.title(
        "Precision-Recall Curve of TEN VAD on TEN-VAD-TestSet",
        fontsize=12,
        color="black",
        fontweight="bold",
    )
    save_path = "PR_Curves.png" #plot save path
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"PR Curves png file saved, save path: {save_path}")