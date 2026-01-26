def convert_label(json_path):
    import json
    with open(json_path,'r',encoding='utf-8') as f:
        ori_label=json.load(f)
    new_label={}
    for audio in ori_label['audios']:
        aid=audio['aid']
        new_label[aid]={
            "duration":audio['duration'],
            "segments":audio['segments']
        }
    with open('wenet_special.json','w',encoding='utf-8') as f:
        json.dump(new_label,f,indent = 4) 

if __name__ == 'main':
    path_to_label="/hpc_stor03/public/shared/data/asr/rawdata/WenetSpeech/data/WenetSpeech.json"
    convert_label(path_to_label)
