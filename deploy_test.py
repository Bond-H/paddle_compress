from paddle.fluid.core import PaddleBuf
from paddle.fluid.core import PaddleDType
from paddle.fluid.core import PaddleTensor
from paddle.fluid.core import AnalysisConfig

from paddle.fluid.core import NativeConfig
from paddle.fluid.core import create_paddle_predictor
import reader

def parse_result(lines, crf_decode,dataset):
    offset_list = crf_decode.lod[0]
    crf_decode = crf_decode.data.int64_data()

    batch_size = len(offset_list) - 1

    batch_out = []
    for sent_index in range(batch_size):
        begin, end = offset_list[sent_index], offset_list[sent_index + 1]
        sent = lines[sent_index]
        tags = [dataset.id2label_dict[str(id)] for id in crf_decode[begin:end]]

        sent_out = []
        tags_out = []
        parital_word = ""
        for ind, tag in enumerate(tags):
            # for the first word
            if parital_word == "":
                parital_word = sent[ind]
                tags_out.append(tag.split('-')[0])
                continue

            # for the beginning of word
            if tag.endswith("-B") or (tag == "O" and tags[ind - 1] != "O"):
                sent_out.append(parital_word)
                tags_out.append(tag.split('-')[0])
                parital_word = sent[ind]
                continue

            parital_word += sent[ind]

        # append the last word, except for len(tags)=0
        if len(sent_out) < len(tags_out):
            sent_out.append(parital_word)

        batch_out.append([sent_out, tags_out])
    return batch_out

class Args():
    model_dir = 'infer_model'
    prog_file = 'output/float/model'
    params_file = 'output/float/weights'
    batch_size = 1
    word_dict_path = "./conf/word.dic"
    label_dict_path = "./conf/tag.dic"
    word_rep_dict_path = "./conf/q2b.dic"


args = Args()
dataset = reader.Dataset(args)


def texts2tensor(texts):
    tensor = PaddleTensor()
    tensor.name = "words"
    lod = [0]
    data = []
    UNK = dataset.word2id_dict[u'OOV']
    for i,text  in enumerate(texts):
        lod.append(len(text)+lod[i])
        data += [dataset.word2id_dict.get(word, UNK) for word in text]
        
    tensor.lod = [lod]
    tensor.shape = [lod[-1], 1]
    tensor.dtype = PaddleDType.INT64
    tensor.data = PaddleBuf(data)

    return tensor

test_data = [u'百度是一家高科技公司', u'中山大学是岭南第一学府']

tensor = texts2tensor(test_data)

config = AnalysisConfig(args.model_dir)
config = AnalysisConfig(args.prog_file, args.params_file)
# config.disable_gpu()
# config.enable_tensorrt_engine()
# config.enable_mkldnn()
# config = NativeConfig()
# config.prog_file = 'infer_model/model.pdmodel'
# config.param_file = 'infer_model/params.pdparams'

# Create PaddlePredictor
predictor = create_paddle_predictor(config)
print(predictor)
outputs = predictor.run([tensor])


result = parse_result(test_data, outputs[0], dataset)
for i, (sent, tags) in enumerate(result):
        result_list = ['(%s, %s)' % (ch, tag) for ch, tag in zip(sent, tags)]
        print(''.join(result_list))
        
        
infer_data = open('data/infer.tsv','r',encoding='utf8').readlines()

from time import time
batch = 100
time1 = time()
for i in range(int(len(infer_data)/batch)):
    inputs = infer_data[i*batch:(i+1)*batch]
    tensor = texts2tensor(inputs)
    outputs = predictor.run([tensor])
    result = parse_result(inputs, outputs[0], dataset)
time2 = time()
print("QPS:",len(infer_data)/(time2-time1))
    
