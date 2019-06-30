# 이미지 잘라서 패치 만들기

def img_transfer(img,imgLabel, bh, bw, no_of_patch):

    h = img.shape[0]
    w = img.shape[1]
    c = img.shape[2]
    ImgArr = np.empty((no_of_patch, bh*bw*3))   ##(패치 개수, 패치 너비x패치 높이x패치 채널<<3채널 칼라>>)의 가상 넘파이 어레이 만듬
    LabelArr = np.empty((no_of_patch, bh*bw*1))  ##(패치 개수 X 패치 너비x패치 높이x패치 채널<<1채널 흑백>>)의 가상 넘파이 어레이 만듬

    # 랜덤으로 패치 가져오기
    for i in range(no_of_patch):
        ih = random.randint(0, h-bh)   ###자르기 시작하고 싶은 부분의 (0,0)이 차지하는 곳에 랜덤 높이
        iw = random.randint(0, w-bw)   ###자르기 시작하고 싶은 부분의 (0,0)이 차지하는 곳에 랜덤 너비
        iArrI = img[ih:ih+bh,iw:iw+bw,:]   ### 이미지 중 자르고 싶은 높이
        iArrL = imgLabel[ih:ih+bh,iw:iw+bw,:]   ###라벨 중 자르고 싶은 높이
        for ci in range(c):
            for bhi in range(bh):
                for bwi in range(bw):
                    ###가상 어레이에 이미지를 자른 부분을 넣는다
                    ImgArr[i][ci*bh*bw + bhi*bw + bwi] = iArrI[bhi][bwi][ci]
                    # 컬러 채널이 0이면 라벨 이미지에 넣는다
                    if ci == 0:
                        LabelArr[i][ci*bh*bw + bhi*bw + bwi] = iArrL[bhi][bwi][ci]

    return ImgArr,LabelArr

def create_dset(patchH, patchW, PatchperImage, settype='train'):
  """
  patchH: Patch height
  patchW: Patch width
  PatchperImage: Number of patches per image
  settype: Can be either train or test
  """
  if settype == 'train':
    Datapath = 'DRIVE/training/images/'
    # 셋트 타입에 따른 패스 변경
    Labelpath = 'DRIVE/training/1st_manual/'
  elif settype == 'test':
    Datapath = 'DRIVE/test/images/'
    # 셋트 타입에 따른 패스 변경
    Labelpath = 'DRIVE/test/1st_manual/'
  else:
    raise ValueError("settype can be either 'test' or 'train'")

  images = torch.DoubleTensor(20*PatchperImage,3*patchH*patchW) # 20 : 총 트레이닝 이미지 개수(여기선 20개)
  labels = torch.DoubleTensor(20*PatchperImage,patchH*patchW)
  t_no = 0
  for img_no in range(20):
      if settype == 'train':
        dp = Datapath + str(img_no+21) + '_training.tif'
        lp = Labelpath + str(img_no+21) + '_manual1.gif'
      elif settype == 'test':
        dp = Datapath + "%02d"%(img_no+1) + '_test.tif'
        lp = Labelpath + "%02d"%(img_no+1) + '_manual1.gif'
      imD = Image.open(dp)
      imD = np.array(imD)

      imL = Image.open(lp)
      imL = np.array(imL)
      imL = np.reshape(imL, (imL.shape[0],imL.shape[1],1))

      imD,imL = img_transfer(imD,imL, patchH, patchW, PatchperImage)
      imD = imD/255.0
      imL = imL/255.0
      for i in range(PatchperImage):
          images[t_no] = torch.from_numpy(imD[i])
          labels[t_no] = torch.from_numpy(imL[i])
          t_no = t_no + 1
  return images, labels
