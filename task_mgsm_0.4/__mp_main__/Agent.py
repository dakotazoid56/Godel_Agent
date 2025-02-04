

class Agent(AgentBase):
    def __init__(agent, api_key=None, goal_prompt_path='goal_prompt.md', key_path='key.env'):
        # Load configurations
        agent.goal_prompt = open(goal_prompt_path, 'r').read()
        agent.goal_task = task_mgsm.MGSM_Task()
        if api_key is None:
            api_key = open(key_path, 'r').read().strip()
        api_key = os.getenv("COHERE_API_KEY")
        agent.client = cohere.ClientV2(api_key=api_key)

        agent.action_functions = [
            {
                "type": "function",
                "function": {
                    "name": "action_display_analysis",
                    "description": "Display an analysis of the current state, including available resources, logic (of solver or other actions) and evaluation feedbacks from the target task, and reasons or plans for the next actions based on this analysis.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "analysis": {
                                "type": "string",
                                "description": "A detailed analysis of the current state, including reasons or plans for the following actions."
                            }
                        },
                        "required": ["analysis"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "action_environment_aware",
                    "description": "Reflect and summarize available resources of the current runtime environment including variables, functions, modules, and external libraries.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "action_read_logic",
                    "description": "Reads the source code of the specified logic (function, method, or class) within a given module.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "module_name": {
                                "type": "string",
                                "description": "The module where the logic resides."
                            },
                            "target_name": {
                                "type": "string",
                                "description": "The name of the function, method, or class to read. If the target_name contains a dot, it refers to a method within a class (e.g., 'Agent.action_call_llm')."
                            }
                        },
                        "required": ["module_name", "target_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "action_adjust_logic",
                    "description": "Modify/Add/Delete the source code of the specified logic (function, method, or class) within a given module to improve task-solving ability or create a tool designed specifically to assist in task-solving efficiently.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "module_name": {
                                "type": "string",
                                "description": "The module where the logic resides."
                            },
                            "target_name": {
                                "type": "string",
                                "description": "The name of the function, method, or class to modify/add/delete. If the target_name contains a dot, it refers to a method within a class."
                            },
                            "new_code": {
                                "type": "string",
                                "description": "The new logic as a string. (Ensure there is no extra indentation in new_code)"
                            },
                            "target_type": {
                                "type": "string",
                                "enum": ["function", "class"],
                                "description": "The type of target."
                            },
                            "operation": {
                                "type": "string",
                                "enum": ["modify", "add", "delete"],
                                "description": "The operation to perform."
                            }
                        },
                        "required": ["module_name", "target_name", "new_code", "target_type", "operation"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "action_run_code",
                    "description": "Execute Python or shell code and capture the output, errors, and return value. (Running python code can get and store objects designed specifically to assist in task-solving efficiently, such as prompts)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code_type": {
                                "type": "string",
                                "enum": ["python", "bash"],
                                "description": "The type of code to execute."
                            },
                            "code": {
                                "type": "string",
                                "description": "The code to execute as a string."
                            }
                        },
                        "required": ["code_type", "code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "action_call_json_format_llm",
                    "description": "Call an external LLM for assistance with gathering insights, refining strategies, correcting errors, and solving complex problems. Output response in JSON format.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "model": {
                                "type": "string",
                                "description": "ID of the model to use."
                            },
                            "messages": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "role": {
                                            "enum": ["system", "assistant", "user"]
                                        },
                                        "content": {"type": "string"}
                                    },
                                    "required": ["role", "content"]
                                },
                                "description": "A list of messages comprising the conversation so far."
                            },
                            "temperature": {
                                "type": "number",
                                "description": "What sampling temperature to use. Higher values will make the output more random, while lower values will make it more focused and deterministic."
                            },
                            "role": {
                                "type": "string",
                                "description": "The role that LLM play."
                            },
                            "return_dict_keys": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                },
                                "description": "An array containing the names of the keys that should be present in the returned dictionary."
                            },
                            "requirements": {
                                "type": "string",
                                "description": "A string that specifies the conditions required to perform a call to the LLM."
                            }
                        },
                        "required": ["model", "messages", "temperature", "role", "return_dict_keys", "requirements"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "action_evaluate_on_task",
                    "description": "Evaluate the current solver on the goal task samples and return the evaluation feedback including valid set accuracy, test set accuray, test sample inputs, model outputs and valid sample answer.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
        ]

        agent.optimize_history = []

    def reinit(agent):
        agent.optimize_history = []
        first_aware_content = action_environment_aware(agent)
        solver_logic = action_read_logic("agent_module", "solver")
        
        print(first_aware_content, end="\n\n")
        print(solver_logic, end="\n\n")

        agent.optimize_history.append({"role": "user", "content": "The logic of solver:\n" + solver_logic})

    def execute_action(agent, actions: typing.Dict):
        """
        Executes the function called by the model and returns the result.
        """
        
        is_reinit = False
        for tool_call in actions['tool_calls']:
            print("tool call:", tool_call, end="\n\n")
            try:
                action_counter[tool_call['function']['name']] += 1
                arguments = json.loads(tool_call['function']['arguments']) if tool_call['function']['arguments'] else {}
                if tool_call['function']['name'] == "action_display_analysis":
                    result = action_display_analysis(**arguments)

                elif tool_call['function']['name'] == "action_environment_aware":
                    result = action_environment_aware(agent, **arguments)

                elif tool_call['function']['name'] == "action_read_logic":
                    result = action_read_logic(**arguments)

                elif tool_call['function']['name'] == "action_adjust_logic":
                    result = action_adjust_logic(**arguments)

                elif tool_call['function']['name'] == "action_run_code":
                    result = action_run_code(**arguments)
                    if arguments.get("code_type", None) == "python" and "self_evolving_agent.reinit()" in arguments.get("code", ""):
                        is_reinit = True
                elif tool_call['function']['name'] == "action_call_llm":
                    result = agent.action_call_llm(**arguments)
                    print(result[0])

                elif tool_call['function']['name'] == 'action_call_json_format_llm':
                    result = agent.action_call_json_format_llm(**arguments)
                    try:
                        print(json.loads(result[0]))
                    except:
                        print(result[0])

                elif tool_call['function']['name'] == "action_evaluate_on_task":
                    result = action_evaluate_on_task(agent.goal_task, functools.partial(solver, agent))
                else:
                    raise ValueError(f"Unknown function name: {tool_call['function']['name']}")

            except Exception as e:
                action_counter["error_handle"] += 1
                exception_stringio = io.StringIO()
                traceback.print_exc(file=exception_stringio)
                result = "Error " + exception_stringio.getvalue()
                exception_stringio.close()

            print("tool call result:\n", result, sep="", end="\n\n")
            if is_reinit:
                break
            agent.optimize_history.append({"role": "tool", 
                                           "content": result, 
                                            "tool_call_id": tool_call['id']})


        print("Action Counter:", action_counter, end='\n\n')
        if action_counter["evolve"] >= 30:
            sys.exit(1)
        print("Agent Evolve", end="\n\n")
        
        agent.evolve()

    def evolve(agent):
        """
        Evolves the agent by prompting the LLM to suggest improvements.
        """
        print('-' * 120)
        action_counter["evolve"] += 1

        tool_call_ids = set()
        remain_optimize_history = []
        for message in agent.optimize_history[-10:]:
            if message["role"] == "assistant" and message["tool_calls"]:
                tool_call_ids = set()
                for tool_call in message["tool_calls"]:
                    tool_call_ids.add(tool_call["id"])
            if message["role"] == "tool" and message["tool_call_id"] not in tool_call_ids:
                print(f"pop item: {message}", end='\n\n')
                continue
            remain_optimize_history.append(message)
        agent.optimize_history = remain_optimize_history

        messages = [{"role": "system", "name": "Principles", "content": agent.goal_prompt}, 
                    {"role": "system", "name": "Environment", "content": action_environment_aware(agent)},
                    *agent.optimize_history]
        try:
            response = agent.action_call_llm(messages=messages, model="command-r7b-12-2024", response_format="text", tools=agent.action_functions, tool_choice="required")

        except Exception as e:
            print(repr(e))
            for message in messages:
                print(message)
            sys.exit(1)
        
        agent.optimize_history.append(response[0])
        agent.execute_action(response[0])
    

    def action_call_json_format_llm(
        agent,
        *,
        messages: typing.List[typing.Dict[str, str]], 
        model: "command-r7b-12-2024",
        temperature: float = 1.0, 
        max_completion_tokens: int = 4096, 
        num_of_response: int = 1,
        role: str = "task solver", 
        return_dict_keys: typing.List[str] = [], 
        requirements: str = "", 
    ):
        system_prompt = (
            f"You are a helpful {role}.\n"
            f"Reply in JSON format, ONLY using the keys {return_dict_keys}.\n"
            f"Requirements:\n{requirements}"
        ).strip()
        _messages = [{"role": "system", "content": system_prompt}, *messages]
        return_dicts = agent.action_call_llm(model=model,
                                    messages=_messages, 
                                    temperature=temperature,
                                    max_completion_tokens=max_completion_tokens,
                                    n=num_of_response,
                                    response_format="json")
        
        for key in return_dict_keys:
            for return_dict in return_dicts:
                if key not in return_dict:
                    return_dict[key] = f"NO {key} IN DICTIONARY"
        return return_dicts
    
    def action_call_llm(
        agent, 
        *,
        #model: typing.Literal["command-r7b-12-2024", "command-r7b-12-2024", "command-r7b-12-2024"] = "command-r7b-12-2024", 
        model:"command-r7b-12-2024",
        messages: typing.List[typing.Dict[str, str]], 
        temperature: float = 1.0, 
        max_completion_tokens: int = 4096, 
        n: int = 1,
        response_format: typing.Literal["text", "json", "json_object"] = "text", 
        tools=None, 
        tool_choice=None,
    ):
        """
        Sends a request to the OpenAI LLM with a system prompt and user message, and returns the response.

        Args:
            agent (Agent): The OpenAI client instance used to interact with the LLM.
            messages (List[Dict[str, str]]): A list of message dictionaries (conversation history).
            response_format (str): The desired format of the LLM's output.
            model (str): Specifies which LLM model to use.
            temperature (float): A float value controlling the randomness of the model's responses. Higher values (e.g., 1.0) increase creativity, while lower values (e.g., 0.1) make the responses more focused and deterministic.
            max_completion_tokens: An integer defining the maximum number of tokens in the completion response, up to 4096.
            n (int): The number of chat completion choices to generate for each input message.

        Returns:
            response (dict): The response from the OpenAI LLM.
        """
        try:
            if response_format == "json":
                response_format = "json_object"
            
            import copy
            messages = copy.deepcopy(messages)
            for message in messages:
                message["content"] = str(message["content"])

            kwargs = {
                "model": model,
                "messages": messages,
                "response_format": {"type": response_format if response_format == "json_object" else "text"}, 
                "temperature": temperature,
                "max_tokens": max_completion_tokens
            }

            if tools is not None:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = tool_choice

            # Make the API call
            #response = agent.client.chat(**kwargs)

            # With Trial Key, Might Exceed 40 API calls per minute, so wait before retrying
            while True:
                try:
                    # Make the API call
                    response = agent.client.chat(**kwargs)
                    break  # Exit loop if successful
                except cohere.errors.TooManyRequestsError:
                    print("Rate limit exceeded. Retrying in 60 seconds...")
                    time.sleep(65)  # Wait for 65 seconds before retrying
                        #cohere.errors.too_many_requests_error.TooManyRequestsError

            def cohere_to_openai_format(cohere_response):
                """
                Convert a Cohere ChatResponse to OpenAI's response format.
                
                Args:
                    cohere_response: Cohere ChatResponse object
                    
                Returns:
                    dict: Response formatted in OpenAI structure
                """
                # Format the message content - handle TextAssistantMessageResponseContentItem
                content = cohere_response.message.content
                if isinstance(content, list):
                    # Combine all text items into a single string
                    content = ''.join(item.text for item in content)
                # Format the message content
                message = {
                    "role": cohere_response.message.role,
                    "content": content,
                    "refusal": None
                }
                
                # Add tool_calls if present
                if cohere_response.message.tool_calls:
                    message["tool_calls"] = [
                        {
                            "id": tool_call.id,
                            "type": tool_call.type,
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments
                            }
                        }
                        for tool_call in cohere_response.message.tool_calls
                    ]
                
                # Create the OpenAI-style response structure
                openai_format = {
                    "id": cohere_response.id,
                    "choices": [
                        {
                            "finish_reason": cohere_response.finish_reason.lower(),
                            "index": 0,
                            "logprobs": cohere_response.logprobs,
                            "message": message
                        }
                    ],
                    "created": None,  # Cohere doesn't provide this
                    "model": None,    # Cohere doesn't provide this
                    "object": "chat.completion",
                    "service_tier": "default",
                    "system_fingerprint": None,
                    "usage": {
                        "completion_tokens": cohere_response.usage.tokens.output_tokens,
                        "prompt_tokens": cohere_response.usage.tokens.input_tokens,
                        "total_tokens": (
                            cohere_response.usage.tokens.input_tokens + 
                            cohere_response.usage.tokens.output_tokens
                        ),
                        "completion_tokens_details": {
                            "accepted_prediction_tokens": 0,
                            "audio_tokens": 0,
                            "reasoning_tokens": 0,
                            "rejected_prediction_tokens": 0
                        },
                        "prompt_tokens_details": {
                            "audio_tokens": 0,
                            "cached_tokens": 0
                        }
                    }
                }
                
                return openai_format
            
            formatted_response = cohere_to_openai_format(response)
            log_model_input_output(kwargs, formatted_response, model)

            def try_parse_json(content):
                try:
                    return json.loads(content)
                except:
                    return {"JSONDecodeError": content}

            if response_format == "text":
                return [choice["message"] for choice in formatted_response["choices"]]
            else:
                return [try_parse_json(choice["message"]["content"]) for choice in formatted_response["choices"]]
    
        except Exception as e:
            raise e
