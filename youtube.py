import torch
from torch.utils.model_zoo import load_url
from scipy.special import expit

import sys
sys.path.append('..')

from blazeface import FaceExtractor, BlazeFace
from architectures import fornet, weights
from isplutils import utils

def video_pred(threshold=0.5, model='EfficientNetAutoAttB4', dataset='DFDC', video_path="notebook/samples/mqzvfufzoq.mp4"):
    """
    Choose an architecture between
    - EfficientNetB4
    - EfficientNetB4ST
    - EfficientNetAutoAttB4
    - EfficientNetAutoAttB4ST
    - Xception
    """
    net_model = model

    """
    Choose a training dataset between
    - DFDC
    - FFPP
    """
    train_db = dataset

    # Setting the parameters
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    face_policy = 'scale'
    face_size = 224

    # Loading the weights
    model_url = weights.weight_url['{:s}_{:s}'.format(net_model, train_db)]
    net = getattr(fornet, net_model)().eval().to(device)
    net.load_state_dict(load_url(model_url, map_location=device, check_hash=True))

    transf = utils.get_transformer(face_policy, face_size, net.get_normalizer(), train=False)

    facedet = BlazeFace().to(device)
    facedet.load_weights("blazeface/blazeface.pth")
    facedet.load_anchors("blazeface/anchors.npy")
    face_extractor = FaceExtractor(facedet=facedet)

    # This part is specific to video processing, which has been removed
    # vid_fake_faces = face_extractor.process_video(video_path)
    # faces_fake_t = torch.stack([transf(image=frame['faces'][0])['image'] for frame in vid_fake_faces if len(frame['faces'])])
    # with torch.no_grad():
    #     faces_fake_pred = net(faces_fake_t.to(device)).cpu().numpy().flatten()

    # if faces_fake_pred.mean() > threshold:
    #     return 'fake', expit(faces_fake_pred.mean())
    # else:
    #     return 'real', expit(faces_fake_pred.mean())

    # Since this file is now only for image processing, we can return a placeholder
    return 'Real', 0.0