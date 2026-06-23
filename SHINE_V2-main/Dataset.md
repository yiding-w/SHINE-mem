# New

return [{
    "context_ids": context_ids,
    "conversation_ids": conversation_ids,
    "labels": labels,
    "context_lengths": context_lengths,
    <!-- "distill": {
        "conversation_ids": distill_conversation_ids,
        "labels": distill_labels,
    }, -->
    "logprob": [value] [idx]
}]

[]  1 trajectory 1 data  len = 1

1 trajectory n chunks 
[{
    "context_ids": context_ids,
    "conversation_ids": conversation_ids,
    "labels": labels,
    "context_lengths": context_lengths,
    <!-- "distill": {
        "conversation_ids": distill_conversation_ids,
        "labels": distill_labels,
    }, -->
    "logprob": [value] [idx]
},
{
    "context_ids": context_ids,
    "conversation_ids": conversation_ids,
    "labels": labels,
    "context_lengths": context_lengths,
    <!-- "distill": {
        "conversation_ids": distill_conversation_ids,
        "labels": distill_labels,
    }, -->
    "logprob": [value] [idx]
},
{
    "context_ids": context_ids,
    "conversation_ids": conversation_ids,
    "labels": labels,
    "context_lengths": context_lengths,
    <!-- "distill": {
        "conversation_ids": distill_conversation_ids,
        "labels": distill_labels,
    }, -->
    "logprob": [value] [idx]
}
]




# Write

1. 1 trajectory 1 data

Shijia 

return [{
    "context_ids": context_ids,
    "conversation_ids": conversation_ids,
    "labels": labels,
    "context_lengths": context_lengths,
    <!-- "distill": {
        "conversation_ids": distill_conversation_ids,
        "labels": distill_labels,
    }, -->
    "logprob": [value] [idx]
}]

10000

context 1,  conversation 2 +
context 2, conversation 3 +
labels ASSISTANT token

2. 1 trajectory multi chunk

[{
    "context_ids": context_ids,
    "conversation_ids": conversation_ids,
    "labels": labels,
    "context_lengths": context_lengths,
    <!-- "distill": {
        "conversation_ids": distill_conversation_ids,
        "labels": distill_labels,
    }, -->
    "logprob": [value] [idx]
},
{
    "context_ids": context_ids,
    "conversation_ids": conversation_ids,
    "labels": labels,
    "context_lengths": context_lengths,
    <!-- "distill": {
        "conversation_ids": distill_conversation_ids,
        "labels": distill_labels,
    }, -->
    "logprob": [value] [idx]
},
{
    "context_ids": context_ids,
    "conversation_ids": conversation_ids,
    "labels": labels,
    "context_lengths": context_lengths,
    <!-- "distill": {
        "conversation_ids": distill_conversation_ids,
        "labels": distill_labels,
    }, -->
    "logprob": [value] [idx]
}
]

1. chunksize 8192
2. to_lora_size 1024

8192 - 1024 = 7000
7000 + 1024 =