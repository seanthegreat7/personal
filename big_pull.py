yolo_number = 0
tar_files = ['962-1.tar',
'963-1.tar','963-2.tar',
'964-1.tar','964-2.tar',
'965-1.tar','965-2.tar',
'966-1.tar',
 '967-1.tar',
 '968-1.tar',
 '969-1.tar',
 '970-1.tar','970-2.tar']
tar_files = [tar_files[yolo_number]]

# ['962-1.tar','962-2.tar',
#  '963-1.tar','963-2.tar',
# '964-1.tar','964-2.tar',
# '965-1.tar','965-2.tar',
# '966-1.tar','966-2.tar',
#  '967-1.tar','967-2.tar',
#  '968-1.tar','968-2.tar',
#  '969-1.tar','969-2.tar',
#  '970-1.tar','970-2.tar']

from ctypes import * 
import tarfile
import os
import pickle
import cv2
import timeit
import numpy as np
import darknet as dn

# dn.set_gpu((yolo_number)%4)
# net = dn.load_net(b"/home/r644352/darknet/cfg/yolov3-test.cfg", b"/home/r644352/darknet/yolov3_final.weights", 0)
# meta = dn.load_meta(b"/home/r644352/darknet/cfg/check.data")
# class_names = ['Name', 'Address', 'pto', 'HandwrittenPayee', 'PrintedPayee', 'HandwrittenAmountInWords', 'PrintedAmountInWords', 'CheckNum', 'Date', 'HandwrittenAmt', 'PrintedAmt', 'Logo', 'Memo', 'Signature', 'PreAuthrorized', 'BankCode','AccountNumber','MICRCheckNum','MICRAmount', 'PayeeName', 'PayeeAddress', 'Fractional']

net = dn.load_net(b"/tmp/ehsan/darknet/cfg/yolov3-test.cfg", b"/tmp/ehsan/darknet/yolov3_final.weights", 0)
meta = dn.load_meta(b"/tmp/ehsan/darknet/cfg/check.data")
class_names = ['Name', 'Address', 'pto', 'HandwrittenPayee', 'PrintedPayee', 'HandwrittenAmountInWords', 'PrintedAmountInWords', 'CheckNum', 'Date', 'HandwrittenAmt', 'PrintedAmt', 'Logo', 'Memo', 'Signature', 'PreAuthrorized', 'BankCode','AccountNumber','MICRCheckNum','MICRAmount', 'PayeeName', 'PayeeAddress', 'Fractional']

# address_in = '/home/r644352/data/'
# address_out = '/data/rqd/fraud/skw/'
address_in = '/tmp/ehsan/'
address_out = '/tmp/ehsan/'
hdfs_path = '/tenants/rqd/fraud/shared/skw_pii/checks/tarfiles/bigpull/'

if not os.path.isdir(address_in):
    os.mkdir(address_in)
# if not os.path.isdir(address_out):
#     os.mkdir(address_out)

t =extracting=encoding=arr_to_imaging=rgbgring=yoloing= 0
transpose=shap_getting=deviding=flattening=copy_and_convert_to_float32= \
ctyping=structuring=image_time=0
big_pull_results = {}

for fn in tar_files:
    ene_to_end_tic = timeit.default_timer()
    print('tar file: ', fn)
    tar_tic = timeit.default_timer()
    tic=timeit.default_timer()
    os.system('kinit a_rqd_fraud_pii_nc -kt ~/a_rqd_fraud_pii_nc.keytab')
    os.system('hdfs dfs -get ' + hdfs_path + fn + ' '+address_in)
    toc=timeit.default_timer()
    print('Got it from HDFS in: ', toc - tic , ' seconds' )
    
    tar = tarfile.open(address_in+fn, 'r|')
    for tiff in tar:
        try:
            if tiff.isfile():
                if t%2000==0 and t!=0:
                    print( 't = ', t)
                    print('extracting average time = ', extracting/t)
                    print('encoding average time = ', encoding/t)
                    print('transpose average time = ', transpose/t)
                    print('shap_getting average time = ', shap_getting/t)
                    print('deviding average time = ', deviding/t)
                    print('flattening average time = ', flattening/t)
                    print('copy_and_convert_to_float32 average time = ', copy_and_convert_to_float32/t)
                    print('ctyping average time = ', ctyping/t)
                    print('structuring average time = ', structuring/t)
                    print('rgbgring average time = ', rgbgring/t)
                    print('yoloing average time = ', yoloing/t)
                    print('one image average time = ', image_time/t)

                image_tic = timeit.default_timer()
                tic = timeit.default_timer()
                raw_flattened = np.frombuffer(tar.extractfile(tiff).read(), dtype=np.uint8)
                toc = timeit.default_timer()
                extracting += (toc -tic)

                tic = timeit.default_timer()
                arr = cv2.imdecode(raw_flattened, cv2.IMREAD_COLOR)
                toc = timeit.default_timer()
                encoding += (toc-tic)


                tic = timeit.default_timer()
                arr = arr.transpose(2,0,1)
                toc = timeit.default_timer()
                transpose += (toc-tic)

                tic = timeit.default_timer()
                c = arr.shape[0]
                h = arr.shape[1]
                w = arr.shape[2]
                toc = timeit.default_timer()
                shap_getting += (toc-tic)

                tic = timeit.default_timer()
                arr = (arr/255.0)
                toc = timeit.default_timer() 
                deviding += (toc-tic)

                tic = timeit.default_timer()
                arr = arr.flatten()
                toc = timeit.default_timer()
                flattening += (toc-tic)

                tic = timeit.default_timer()
                a = np.copy(arr).astype(np.float32)
                toc = timeit.default_timer()
                copy_and_convert_to_float32 += (toc-tic)

                tic = timeit.default_timer()
                new_arr = np.ctypeslib.as_ctypes(a)
                toc = timeit.default_timer()
                ctyping += (toc-tic)   

                tic = timeit.default_timer()
                im = dn.IMAGE(w,h,c,new_arr)
                toc = timeit.default_timer()
                structuring += (toc-tic)

                tic = timeit.default_timer()
                dn.rgbgr_image(im)
                toc = timeit.default_timer()
                rgbgring += (toc-tic)

                tic = timeit.default_timer()
                big_pull_results[str(tiff).split("'")[1].split('/')[-1]] = dn.detect(net, meta, im)
                toc = timeit.default_timer()
                yoloing += toc - tic
                t += 1
                image_toc = timeit.default_timer()
                image_time += image_toc - image_tic
#                 if t == 301:
#                     break
        except Exception as e:
            print('problem with', infile, 'in ', fn, 'because of ', e )
            continue
    tar.close()
    tar_toc = timeit.default_timer()
    tic = timeit.default_timer()
    os.system('rm '+ address_in+fn)
    toc = timeit.default_timer()
    print('Deleting tar file took: ', toc-tic, ' seconds')
    print( t , ' files processed in ',tar_toc - tar_tic, ' seconds \n average time per file = ', (tar_toc - tar_tic)/t)
    print('Start pickling')
    tic=timeit.default_timer()
    with open(address_in+fn+'.pkl', 'wb') as f:
        pickle.dump(big_pull_results, f)
    toc=timeit.default_timer()
    print('done pickling in: ', toc - tic)
    os.system('kinit a_rqd_fraud_pii_nc -kt ~/a_rqd_fraud_pii_nc.keytab')
    print('sending it to HDFS...')
    tic=timeit.default_timer()
    os.system('hdfs dfs -copyFromLocal -f '+address_out+ fn+'.pkl' + ' ' + '"/tenants/rqd/fraud/shared/skw_pii/users/ehsan/+%s.pkl"' % fn)
    toc=timeit.default_timer()
    print('sent to HDFS in: ', toc - tic , ' seconds')
    end_to_end_toc = timeit.default_timer()
    print('End to end time: ', end_to_end_toc - end_to_end_tic, ' seconds \n average time per file: ', (end_to_end_toc - end_to_end_tic)/t)
