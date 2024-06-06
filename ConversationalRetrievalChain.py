from BaseConversationalRetrievalChain import *
from config import DEFAULT_CONFIG
class ConversationalRetrievalChain(BaseConversationalRetrievalChain):
    
    retriever: BaseRetriever
    rephrasequestion: Optional[bool] = DEFAULT_CONFIG['rephrase']
    """Retriever to use to fetch documents."""
    max_tokens_limit: Optional[int] = None

    
    def _get_docs(
        self,
        question: str,
        inputs: Dict[str, Any],
        *,
        run_manager: CallbackManagerForChainRun,
    ) -> List[Document]:
        """Get docs."""
        
        docs = self.retriever.get_relevant_documents(
            question
        )
        return docs

    async def _aget_docs(
        self,
        question: str,
        inputs: Dict[str, Any],
        *,
        run_manager: AsyncCallbackManagerForChainRun,
    ) -> List[Document]:
        """Get docs."""
       
        docs = await self.retriever.aget_relevant_documents(
            question
        )
        return docs

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        retriever: BaseRetriever,
        condense_question_prompt: BasePromptTemplate = CONDENSE_QUESTION_PROMPT,
        chain_type: str = "stuff",
        rephrasequestion: Optional[bool] = DEFAULT_CONFIG['rephrase'],
        verbose: bool = False,
        condense_question_llm: Optional[BaseLanguageModel] = None,
        combine_docs_chain_kwargs: Optional[Dict] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> BaseConversationalRetrievalChain:
        
        combine_docs_chain_kwargs = combine_docs_chain_kwargs or {}
        doc_chain = load_qa_chain(
            llm,
            chain_type=chain_type,
            verbose=verbose,
            callbacks=callbacks,
            **combine_docs_chain_kwargs,
        )
        
        
        if rephrasequestion  : 
            _llm = condense_question_llm or llm
            condense_question_chain = LLMChain(
                llm=_llm,
                prompt=condense_question_prompt,
                verbose=verbose,
                callbacks=callbacks,
            )
            return cls(
                retriever=retriever,
                combine_docs_chain=doc_chain,
                question_generator=condense_question_chain,
                callbacks=callbacks,
                **kwargs,
            )
            
        else:
            
            #print("rephrasequestion = " , rephrasequestion)
            return cls(
                retriever=retriever,
                combine_docs_chain=doc_chain,
                
                callbacks=callbacks,
                **kwargs,
            )
        
        
            



