#!/usr/bin/python
#
# Cityscapes labels
#

from collections import namedtuple


#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for you approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'void'                  ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'sky'                   ,  1 ,      9 , 'void'            , 0       , False        , True         , (  70,130,180) ),
    Label(  'building'              ,  2 ,      2 , 'void'            , 0       , False        , True         , (  70,70,70) ),
    Label(  'road'                  ,  3 ,      0 , 'void'            , 0       , False        , True         , (  128,64,128) ),
    Label(  'sidewalk'              ,  4 ,      1 , 'void'            , 0       , False        , True         , (  244,35,232) ),
    Label(  'fence'                 ,  5 ,      4 , 'void'            , 0       , False        , True         , (  64,64,128) ),
    Label(  'vegetation'            ,  6 ,      8 , 'void'            , 0       , False        , True         , (  107,142,35) ),
    Label(  'pole'                  ,  7 ,      5 , 'flat'            , 1       , False        , False        , (  153,153,153) ),
    Label(  'car'                   ,  8 ,      12 , 'flat'            , 1       , False        , False        , ( 0,0,142) ),
    Label(  'traffic sign'          ,  9 ,      7 , 'flat'            , 1       , False        , True         , (  220,220,0) ),
    Label(  'pedestrian'            , 10 ,      10 , 'flat'            , 1       , False        , True         , ( 220,20,60) ),
    Label(  'bicycle'               , 11 ,      15 , 'construction'    , 2       , False        , False        , ( 119,11,32) ),
    Label(  'motorcycle'            , 12 ,      14 , 'construction'    , 2       , False        , False        , ( 0,0,230) ),
    Label(  'parking-slot'          , 13 ,      255 , 'construction'    , 2       , False        , False        , (250,170,160) ),
    Label(  'road-work'             , 14 ,      255 , 'construction'    , 2       , False        , True         , (128,64,64) ),
    Label(  'traffic light'         , 15 ,      6 , 'construction'    , 2       , False        , True         , (250,170,30) ),
    Label(  'terrain'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (152,251,152) ),
    Label(  'rider'                 , 17 ,      11 , 'object'          , 3       , False        , False        , (255,0,0) ),
    Label(  'truck'                 , 18 ,      255 , 'object'          , 3       , False        , True         , (0,0,70) ),
    Label(  'bus'                   , 19 ,      13 , 'object'          , 3       , False        , False        , (0,60,100) ),
    Label(  'train'                 , 20 ,      255 , 'object'          , 3       , False        , False        , (0,80,100) ),
    Label(  'wall'                  , 21 ,      3 , 'nature'          , 4       , False        , False        , (102,102,156) ),
    Label(  'lanemarking'           , 22 ,      0 , 'nature'          , 4       , False        , False        , (102,102,156) ),

]


#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

# name to label object
name2label      = { label.name    : label for label in labels           }
# id to label object
id2label        = { label.id      : label for label in labels           }
# trainId to label object
trainId2label   = { label.trainId : label for label in reversed(labels) }
# category to list of label objects
category2labels = {}
for label in labels:
    category = label.category
    if category in category2labels:
        category2labels[category].append(label)
    else:
        category2labels[category] = [label]

#--------------------------------------------------------------------------------
# Assure single instance name
#--------------------------------------------------------------------------------

# returns the label name that describes a single instance (if possible)
# e.g.     input     |   output
#        ----------------------
#          car       |   car
#          cargroup  |   car
#          foo       |   None
#          foogroup  |   None
#          skygroup  |   None
def assureSingleInstanceName( name ):
    # if the name is known, it is not a group
    if name in name2label:
        return name
    # test if the name actually denotes a group
    if not name.endswith("group"):
        return None
    # remove group
    name = name[:-len("group")]
    # test if the new name exists
    if not name in name2label:
        return None
    # test if the new name denotes a label that actually has instances
    if not name2label[name].hasInstances:
        return None
    # all good then
    return name

#--------------------------------------------------------------------------------
# Main for testing
#--------------------------------------------------------------------------------

# just a dummy main
if __name__ == "__main__":
    # Print all the labels
    print("List of cityscapes labels:")
    print("")
    print("    {:>21} | {:>3} | {:>7} | {:>14} | {:>10} | {:>12} | {:>12}".format( 'name', 'id', 'trainId', 'category', 'categoryId', 'hasInstances', 'ignoreInEval' ))
    print("    " + ('-' * 98))
    for label in labels:
        print("    {:>21} | {:>3} | {:>7} | {:>14} | {:>10} | {:>12} | {:>12}".format( label.name, label.id, label.trainId, label.category, label.categoryId, label.hasInstances, label.ignoreInEval ))
    print("")

    print("Example usages:")

    # Map from name to label
    name = 'car'
    id   = name2label[name].id
    print("ID of label '{name}': {id}".format( name=name, id=id ))

    # Map from ID to label
    category = id2label[id].category
    print("Category of label with ID '{id}': {category}".format( id=id, category=category ))

    # Map from trainID to label
    trainId = 0
    name = trainId2label[trainId].name
    print("Name of label with trainID '{id}': {name}".format( id=trainId, name=name ))
