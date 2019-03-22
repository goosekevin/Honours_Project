# Semantic Segmentation
# Code by GunhoChoi

from testUnet2 import *



import matplotlib.pyplot as plt

def noisy(noise_typ,image):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 15
        sigma = var**0.5
        print(sigma)
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss + gauss + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy


def getTrainingData():

    transform = transforms.Compose([transforms.ToTensor(),])

    img_dir = "cutpic\input"
    gt_dir = "cutpic\gtruth"
    data_path1 = os.path.join(img_dir,'*.png')
    data_path2 = os.path.join(gt_dir,'*.png')
    files1 = sorted(glob.glob(data_path1), key=os.path.getmtime)
    files2 = sorted(glob.glob(data_path2), key=os.path.getmtime)
    data_input = []
    data_label = []

    for f1 in files1:
        img = cv2.imread(f1)
        img = transform(img)
        img = img.unsqueeze(0)
        data_input.append(img)

    for f2 in files2:
        img = cv2.imread(f2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = torch.from_numpy(img.astype(int))
        img = img.long()
        img = img.unsqueeze(0)
        data_label.append(img)   

    tuple_list = []
    for x in range(len(data_input)):
        pair = (data_input[x], data_label[x])
        tuple_list.append(pair)  

    return tuple_list

def getTestData(img_size=320):
    transform = transforms.Compose([transforms.ToTensor(),])

    img_dir = "cutpic\\test"

    data_path1 = os.path.join(img_dir,'*.png')
    files1 = sorted(glob.glob(data_path1), key=os.path.getmtime)
    data_input = []

    for f1 in files1:
        img = cv2.imread(f1)

        img = img[0:img_size,0:img_size,:]
        print(img.shape)
        cv2.imwrite("result\original.png", img)
        #scipy.misc.imsave('test\pic\gradient.png', arr_gtruth)
        img = noisy("gauss", img)
        cv2.imwrite("result\\noisy.png", img)
        img = transform(img)
        img = img.float()
        img = img.unsqueeze(0)
        data_input.append(img)

    return data_input

def getBatchData(tuple_list, batch_size = 1):
    batch_tuple_list = []
    shuffle(tuple_list)
    x = 0
    while x < len(tuple_list):
        arg1 = tuple_list[x][0]
        arg2 = tuple_list[x][1]
        x += 1
        for i in range(batch_size-1):
            if x >= len(tuple_list):
                break

            arg3 = tuple_list[x][0]
            arg1 = torch.cat((arg1, arg3),0)

            arg4 = tuple_list[x][1]
            arg2 = torch.cat((arg2, arg4),0)

            x +=1

        pair = (arg1, arg2)
        batch_tuple_list.append(pair)

    return batch_tuple_list

def displayGenerated(gen, i, k, mode="standard"):
    bat = gen.shape[0]
    l = gen.shape[2]
    w = gen.shape[3]
    gen = gen.cpu().numpy()

    for z in range(bat):
        dis = np.zeros((l, w, 3), dtype=np.uint8)
        for x in range(l):
            for y in range(w):
                if gen.item((z, 0, x, y)) == 0:
                    dis[x,y,0] = 255
                    if mode == "difference":
                        dis[x,y,1] = 255
                        dis[x,y,2] = 255
                if gen.item((z, 0, x, y)) == 1:
                    dis[x,y,1] = 255
                if gen.item((z, 0, x, y)) == 2:
                    dis[x,y,2] = 255
        scipy.misc.imsave('result\\argmax\\argmax_{}_{}_{}.png'.format(i,k,z), dis)

def displayTruth(truth, i, k):
    bat = truth.shape[0]
    l = truth.shape[1]
    w = truth.shape[2]
    
    truth = truth.cpu().numpy()
    for z in range(bat):
        dis = np.zeros((l, w), dtype=np.uint8)
        for x in range(l):
            for y in range(w):
                if truth.item((z,x,y)) == 0:
                    dis[x,y] = 0
                if truth.item((z,x,y)) == 1:
                    dis[x,y] = 100
                if truth.item((z,x,y)) == 2:
                    dis[x,y] = 200
        scipy.misc.imsave('result\\truth_{}_{}_{}.png'.format(i,k,z), dis)


if __name__ == '__main__':

    # hyperparameters
    mode = "test"
    batch_size = 2
    img_size = 320
    lr = 0.0002
    epoch = 1

    # initiate Generator

    generator = nn.DataParallel(UnetGenerator(3,3,64),device_ids=[i for i in range(1)]).cuda()

    # load pretrained model

    try:
        generator = torch.load('./model/{}.pkl'.format("unet"))
        print("\n--------model restored--------\n")
    except:
        print("\n--------model not restored--------\n")
        pass

    # loss function & optimizer

    recon_loss_func = nn.CrossEntropyLoss()
    gen_optimizer = torch.optim.Adam(generator.parameters(),lr=lr)

    # Training

    if mode == "train":

        train_data = getTrainingData()

        file = open('./{}_CE_loss.txt'.format("unet"), 'w')
        for i in range(epoch):

            total_loss = 0
            batchData = getBatchData(train_data, batch_size=batch_size)
            for k in range(len(batchData)):
                image = batchData[k][0] 
                gtruth = batchData[k][1]
                
                gen_optimizer.zero_grad()

                x = Variable(image).cuda(0)
                y_ = Variable(gtruth).cuda(0)
                y = generator.forward(x)
                
                loss = recon_loss_func(y,y_)
                total_loss += loss
                loss.backward()
                gen_optimizer.step()

                

                if k % 100 == 0:
                    print(i)
                    print(loss)
                    y_argmax = torch.argmax(y, dim=1)
                    y_argmax = y_argmax.unsqueeze(1)
                    displayGenerated(y_argmax, i, k)


                    v_utils.save_image(x.cpu().data,"./result/original_image_{}_{}.png".format(i,k))
                    displayTruth(y_, i, k)
                    v_utils.save_image(y.cpu().data,"./result/gen_image_{}_{}.png".format(i,k))
                    #torch.save(generator,'./model/{}.pkl'.format("unet"))
   
            ave_loss = total_loss / len(batchData)
            file.write(str(ave_loss)+"\n")
            print("average loss for batch")
            print(ave_loss)


    #Testing

    if mode == "test":

        test_data = getTestData()
        
        comparison = False
        i = 0
        while i < 3:

            image = test_data[i]
            
            #print(torch.sum(image).item())
            x = Variable(image).cuda(0)
            y = generator.forward(x, mode=mode, comparison=comparison)
            
            if comparison == False:
                comparison = True
                v_utils.save_image(y.cpu().data,"./result/gen_image_{}.png".format(i))
            else:
                v_utils.save_image(y.cpu().data,"./result/gen_image_{}.png".format(i))
                y_argmax = torch.argmax(y, dim=1)
                y_argmax = y_argmax.unsqueeze(1)
                displayGenerated(y_argmax,i,0, mode="difference")
                comparison = False
            i +=1

